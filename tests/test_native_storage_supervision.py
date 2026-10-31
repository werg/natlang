import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('native_storage_supervision',
    Path(__file__).resolve().parents[1] / 'scripts/native_storage_supervision.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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
