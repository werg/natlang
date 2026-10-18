import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.coreweave import teacher_incremental_export as export
from test_teacher_primary_overlap import publisher_overlap_fixture


def adopted_fixture(*, closed=False, bad_birth=False):
    publisher, _, remote, state, store, _ = publisher_overlap_fixture()
    original_owner = copy.deepcopy(state['owner'])
    slot = state.pop('primary_prefetch')
    state['primary_launches'] = {'23': copy.deepcopy(slot)}
    state.update(active_batch=23, owner=copy.deepcopy(slot['owner']))
    state['source_sequence'].pop('prefetch')
    if closed:
        boundary = store('/work/control/boundaries/batch-0023.json', {'batch': 23})
        receipt = store('/work/control/source-sequence/closed/batch-0023.json', {
            'batch': 23, 'ordinal': 1,
            'started': state['source_sequence']['started']['23'], 'boundary': boundary})
        state['source_sequence']['closed'] = {'23': receipt}
        state.update(active_batch=24, owner={'pid': 44, 'start_ticks': '999', 'boot_id': 'boot'})
        store(slot['phase_path'], {'phase': 'batch_complete', 'batch': 23})
    store('/work/control/state.json', state)

    def execute(argv, **_):
        if argv[0] == 'cat':
            value = 'boot\n'
        elif 'glob' in argv[2]:
            value = json.dumps({'queues': ['queue-worker-0.jsonl'], 'journals': ['journal-worker-0.jsonl']})
        else:
            owner = original_owner if argv[-1] == '22' else slot['owner']
            feeder = '/work/control/feed.py' if argv[-1] == '22' else slot['feeder']['path']
            assert argv[-5:] == [str(owner['pid']), owner['start_ticks'], 'boot', feeder, argv[-1]]
            if closed:
                value = json.dumps({'valid': False, 'missing': True, 'boot_id': 'boot'})
            elif bad_birth:
                value = json.dumps({'valid': False, 'identity': {**owner, 'start_ticks': 'changed'}})
            else:
                value = json.dumps({'valid': True})
        return SimpleNamespace(result=lambda **_: SimpleNamespace(returncode=0, stdout=value, stderr=''))

    return publisher, SimpleNamespace(exec=execute), remote, state, store, slot


def test_requested_lookahead_can_become_the_exact_live_primary():
    publisher, sandbox, remote, state, _, slot = adopted_fixture()
    value = export.discover_guardian_batch(publisher, sandbox, remote.__getitem__, batch_slot=23)
    assert value['batch'] == 23
    assert value['queue_path'].endswith('batch-0023/queue.jsonl')
    assert value['guardian_state']['owner'] == slot['owner'] == state['owner']
    assert value['identity']['runtime_path'] == slot['task_runtime']


def test_requested_slot_closed_after_adoption_needs_no_live_old_process():
    publisher, sandbox, remote, _, _, _ = adopted_fixture(closed=True)
    value = export.discover_guardian_batch(publisher, sandbox, remote.__getitem__, batch_slot=23)
    assert value['draining'] and value['batch'] == 23
    assert value['closed_boundaries'][0]['path'].endswith('/boundaries/batch-0023.json')
    assert 'queue_path' not in value  # no new live export custody from draining


@pytest.mark.parametrize('tamper', ['owner', 'birth', 'runtime', 'source', 'future', 'history', 'closed', 'closed_started'])
def test_promotion_does_not_release_wrong_or_unowned_custody(tamper):
    publisher, sandbox, remote, state, store, slot = adopted_fixture(
        closed=tamper in ('closed', 'closed_started'), bad_birth=tamper == 'birth')
    requested = 23
    if tamper == 'owner':
        state['owner'] = {**state['owner'], 'pid': 99}
    elif tamper == 'runtime':
        state['primary_launches']['23']['task_runtime'] = '/wrong/runtime'
    elif tamper == 'source':
        remote['/work/generation/evidence-collections-v2/batch-0023/source-cases.jsonl'] = b'wrong source'
    elif tamper == 'future':
        requested = 24
    elif tamper == 'history':
        state['primary_launches'] = {}
    elif tamper == 'closed':
        state['source_sequence']['closed'] = {}
    elif tamper == 'closed_started':
        ref = state['source_sequence']['closed']['23']
        value = json.loads(remote[ref['path']]); value['started'] = {'path': '/wrong/started'}
        state['source_sequence']['closed']['23'] = store(ref['path'], value)
    store('/work/control/state.json', state)
    with pytest.raises(ValueError):
        export.discover_guardian_batch(publisher, sandbox, remote.__getitem__, batch_slot=requested)


def test_real_service_scan_survives_promotion_after_original_export(tmp_path, monkeypatch):
    from scripts.coreweave import source_archive
    publisher, sandbox, remote, adopted, store, slot = adopted_fixture()
    before = copy.deepcopy(adopted)
    before.update(active_batch=22, owner={'pid': 42, 'start_ticks': '777', 'boot_id': 'boot'}, primary_prefetch=copy.deepcopy(slot))
    before.pop('primary_launches')
    before['source_sequence']['prefetch'] = {'batch': 23, 'intent': slot['selection_intent']}
    store('/work/control/state.json', before)
    publisher.update(directory_root=str(tmp_path/'exports'), chunk_size=1)
    path = tmp_path/'publisher.json'; path.write_text(json.dumps(publisher))
    for batch in (22, 23):
        home = f'/work/generation/evidence-collections-v2/batch-{batch:04d}'
        entry = json.loads(remote[home+'/queue.jsonl'])
        payload = json.dumps({'trajectory': [{'task': entry['task_id'], 'batch': batch}]}).encode()+b'\n'
        remote[entry['output']] = payload
        store(entry['output']+'.manifest.json', {'version': 'natlang.teacher_batch.native/1',
            'source': entry['source'], 'source_sha256': hashlib.sha256(remote[entry['source']]).hexdigest(),
            'range': {'start': 0, 'count': 1}, 'completed': 1, 'missing': [],
            'output_sha256': hashlib.sha256(payload).hexdigest()})
        finish = {'event': 'finish', 'key': entry['key'], 'output_accounting': {
            'version': 'natlang.supervisor_output_accounting/2', 'manifest_valid': True,
            'manifest_errors': [], 'complete': True, 'expected_jobs': 1, 'exact_result_rows': 1,
            'output_sha256': hashlib.sha256(payload).hexdigest(), 'job_states': [{'state': 'result'}]}}
        remote[home+'/journal-worker-0.jsonl'] = b''.join(json.dumps(row).encode()+b'\n' for row in (
            {'event': 'start', 'key': entry['key']}, finish))
    monkeypatch.setattr(export, '_cwctl_module', lambda: SimpleNamespace(load_env_file=lambda: None))
    monkeypatch.setattr(export, 'scoped_sdk_session', lambda _: (sandbox, remote.__getitem__))
    pack = source_archive.pack
    monkeypatch.setattr(source_archive, 'pack', lambda *a, **k: pack(*a, **{**k, 'reserve_bytes': 0}))
    uploads = []

    def publish(tranche, *, exporter, role, receipt_path):
        raw = Path(tranche['archive_receipt']['archive']).read_bytes()
        identity = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({'private': True, 'run': 'test/run', 'name': 'archive',
            'identity': identity, 'independent_verification': identity, 'verified_at': 123.0}))
        uploads.append(tranche['batch'])
        if tranche['batch'] == 22:
            store('/work/control/state.json', adopted)
        return export.accept_private_receipt(receipt_path, tranche)

    monkeypatch.setattr(export, 'export_private_tranche', publish)
    result = export.run_guardian_export_service(path, poll_seconds=0, max_polls=2, allowed_spool_root=tmp_path)
    assert uploads == [22, 23]
    assert result['active_identity']['batch'] == 23
    for batch in (22, 23):
        frontier = json.loads((tmp_path/f'exports/batch-{batch:04d}/frontier.json').read_text())
        assert frontier['completed_keys'] == [f'task-{batch}']
