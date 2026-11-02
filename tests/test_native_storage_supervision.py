import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('native_storage_supervision',
    Path(__file__).resolve().parents[1] / 'scripts/native_storage_supervision.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('entrypoint,validate,expected', [
    ('natlang_neuralese.train.native_state_sft', False, True),
    ('natlang_neuralese.rl.native_stage', False, True),
    ('natlang_neuralese.train.native_state_sft', True, False),
    ('unrelated.module', False, False),
])
def test_monitor_finds_real_sft_and_rl_writers(tmp_path, entrypoint, validate, expected):
    proc = tmp_path / 'proc' / '123'
    proc.mkdir(parents=True)
    out = tmp_path / 'learner'
    out.mkdir()
    args = ['python', '-m', entrypoint, '--out', str(out)]
    # A string in another flag must not establish a trainer entrypoint.
    args += ['--label', 'natlang_neuralese.train.native_state_sft']
    if validate:
        args.append('--validate-continuation-only')
    (proc / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in args))
    (proc / 'environ').write_bytes(b'NATLANG_LEDGER_UNIT=real-learner.service\0')
    (out / 'progress.json').write_text(json.dumps({'global_step': 70}))
    found = module.current_native_writers(tmp_path / 'proc')
    assert bool(found) is expected
    if expected:
        assert found[0]['out'] == str(out)
        assert found[0]['unit'] == 'real-learner.service'
        assert found[0]['progress']['global_step'] == 70
        assert found[0]['reservation']['state'] == 'not-yet-reserved'


def test_monitor_distinguishes_missing_reservation_from_authenticated_inode(tmp_path):
    slot = tmp_path / 'checkpoint.pt'
    slot.write_bytes(b'last committed model')
    assert module.native_reservation_status(slot)['state'] == 'not-yet-reserved'
    pending = tmp_path / 'checkpoint.pt.reserved-pending'
    pending.write_bytes(b'allocated future checkpoint')
    st = pending.stat()
    receipt = tmp_path / 'checkpoint.pt.reservation.json'
    receipt.write_text(json.dumps({'schema': 'natlang.checkpoint-reservation/1',
                                  'device': st.st_dev, 'inode': st.st_ino}))
    assert module.native_reservation_status(slot)['protected']
    receipt.write_text(json.dumps({'schema': 'natlang.checkpoint-reservation/1',
                                  'device': st.st_dev, 'inode': st.st_ino + 1}))
    assert module.native_reservation_status(slot)['state'] == 'invalid-reservation'


def test_monitor_never_counts_durable_checkpoint_as_reserved_space(tmp_path):
    import os
    slot = tmp_path / 'checkpoint.pt'
    slot.write_bytes(b'only durable copy')
    pending = tmp_path / 'checkpoint.pt.reserved-pending'
    os.link(slot, pending)
    st = pending.stat()
    (tmp_path / 'checkpoint.pt.reservation.json').write_text(json.dumps({
        'schema': 'natlang.checkpoint-reservation/1', 'device': st.st_dev, 'inode': st.st_ino}))
    assert not module.native_reservation_status(slot)['protected']


def test_sparse_pending_length_does_not_establish_allocated_reservation(tmp_path):
    slot = tmp_path / 'checkpoint.pt'
    pending = tmp_path / 'checkpoint.pt.reserved-pending'
    with pending.open('wb') as stream:
        stream.truncate(1024 * 1024)
    st = pending.stat()
    (tmp_path / 'checkpoint.pt.reservation.json').write_text(json.dumps({
        'schema': 'natlang.checkpoint-reservation/1', 'device': st.st_dev, 'inode': st.st_ino}))
    observed = module.native_reservation_status(slot)
    assert observed['state'] == 'partially-allocated' and not observed['protected']


def test_monitor_reports_both_slots_and_bound_roles(tmp_path):
    slot = tmp_path / 'checkpoint.pt'
    identities = {}
    for suffix, role in (('', 'writer'), ('.alternate', 'next')):
        pending = slot.with_name(slot.name + suffix + '.reserved-pending')
        pending.write_bytes(b'allocated')
        st = pending.stat()
        receipt = slot.with_name(slot.name + suffix + '.reservation.json')
        receipt.write_text(json.dumps({'schema': 'natlang.checkpoint-reservation/1',
                                       'device': st.st_dev, 'inode': st.st_ino}))
        identities[role] = {'path': str(pending), 'device': st.st_dev, 'inode': st.st_ino}
    slot.with_name(slot.name + '.reservation-roles.json').write_text(json.dumps({
        'schema': 'natlang.checkpoint-reservation-roles/1', 'pid': 1, **identities}))
    observed = module.native_reservation_status(slot)
    assert observed['slots']['primary']['protected'] and observed['slots']['alternate']['protected']
    assert observed['roles']['writer'] == identities['writer']
    assert observed['roles']['next'] == identities['next']
    identities['next']['inode'] += 1
    slot.with_name(slot.name + '.reservation-roles.json').write_text(json.dumps({
        'schema': 'natlang.checkpoint-reservation-roles/1', 'pid': 1, **identities}))
    assert module.native_reservation_status(slot)['roles']['state'] == 'stale-or-invalid'
