import hashlib
import json
from pathlib import Path

import pytest

from scripts.coreweave import source_archive
from scripts.coreweave.teacher_incremental_export import (
    accept_private_receipt, completed_entries, export_finished_tranches,
    prune_verified_local_tranche, seal_boundary, seal_tranche,
)


@pytest.fixture(autouse=True)
def small_archive_spool_reserve(monkeypatch):
    """Unit fixtures are tiny; leave the production 20 GiB reserve untouched."""
    original = source_archive.pack
    def pack(source, spool, name, **kwargs):
        kwargs['reserve_bytes'] = 0
        return original(source, spool, name, **kwargs)
    monkeypatch.setattr(source_archive, 'pack', pack)


def _fixture():
    entries = [
        {'key': 's7-task-000-00', 'index': 0, 'count': 1, 'sample': 0, 'seed': 3,
         'source': '/batch/source-cases.jsonl', 'output': '/batch/out/00/results.jsonl',
         'jobs': '/batch/out/00/jobs', 'task_id': 'task-a'},
        {'key': 's7-task-001-00', 'index': 1, 'count': 1, 'sample': 0, 'seed': 4,
         'source': '/batch/source-cases.jsonl', 'output': '/batch/out/01/results.jsonl',
         'jobs': '/batch/out/01/jobs', 'task_id': 'task-b'},
    ]
    queue = b''.join(json.dumps(row).encode() + b'\n' for row in entries)
    finish = lambda key: {'event': 'finish', 'key': key, 'output_accounting': {
        'version': 'natlang.supervisor_output_accounting/2', 'manifest_valid': True,
        'manifest_errors': [], 'complete': True, 'expected_jobs': 1,
        'exact_result_rows': 1, 'output_sha256': 'a'*64,
        'job_states': [{'state': 'result'}]}}
    journal = b''.join(json.dumps({'event': 'start', 'key': row['key']}).encode() + b'\n'
                       + json.dumps(finish(row['key'])).encode() + b'\n' for row in entries)
    cases = b'{"id":"task-a"}\n{"id":"task-b"}\n'
    teacher_source = {'schema': 'natlang.sandbox-strong-teacher-collection/1', 'batch': 19,
        'source_cases_sha256': hashlib.sha256(cases).hexdigest(), 'actor': {'sha256': 'a'*64},
        'actual_teacher_execution_identity': {'sha256': 'b'*64}, 'actual_task_runtime': {'sha256': 'c'*64},
        'collection_controller': {'sha256': 'd'*64}, 'runtime_source_receipt': {'sha256': 'e'*64},
        'teacher_configuration_source': {'sha256': 'f'*64}, 'source_provenance': {'sha256': '0'*64}}
    context = {'teacher-source.json': json.dumps(teacher_source).encode(), 'queue.jsonl': queue,
               'source-cases.jsonl': cases,
               'repair-queue.jsonl': b''}
    outputs = {}
    for entry in entries:
        payload = json.dumps({'trajectory': [{'task': entry['task_id']}]}).encode() + b'\n'
        manifest = json.dumps({'version': 'natlang.teacher_batch.native/1',
            'source': entry['source'], 'source_sha256': hashlib.sha256(context['source-cases.jsonl']).hexdigest(),
            'range': {'start': entry['index'], 'count': 1}, 'completed': 1, 'missing': [],
            'output_sha256': hashlib.sha256(payload).hexdigest()}).encode()
        event = finish(entry['key'])
        event['output_accounting'].update(exact_result_rows=1,
                                          output_sha256=hashlib.sha256(payload).hexdigest())
        # Keep the output accounting digest in the queue-selected journal event.
        journal_events = [json.loads(line) for line in journal.splitlines()]
        journal_events = [event if row.get('key') == entry['key'] and row.get('event') == 'finish' else row
                          for row in journal_events]
        journal = b''.join(json.dumps(row).encode()+b'\n' for row in journal_events)
        outputs[entry['key']] = payload, manifest
    return entries, queue, journal, context, outputs, finish


def test_completed_tranche_frontier_is_ordered_terminal_and_resume_safe():
    entries, queue, journal, context, _, finish = _fixture()
    event = finish(entries[0]['key']); event['output_accounting']['output_sha256'] = 'a'*64
    partial = json.dumps(event).encode() + b'\n'
    assert [row['key'] for row in completed_entries(queue, partial)] == [entries[0]['key']]
    assert completed_entries(queue, partial, after={entries[0]['key']}) == []
    unresolved = json.loads(partial)
    unresolved['output_accounting']['complete'] = False
    assert completed_entries(queue, json.dumps(unresolved).encode() + b'\n') == []


def test_jsonl_parser_preserves_unicode_line_separators_inside_json_strings():
    entries, _, journal, _, _, _ = _fixture()
    first = dict(entries[0], note='line\u2028separator\u2029is string content')
    raw = json.dumps(first, ensure_ascii=False).encode('utf-8') + b'\n'
    finish = json.loads(journal.split(b'\n')[1])
    result = completed_entries(raw, json.dumps(finish, ensure_ascii=False).encode() + b'\n')
    assert result[0]['note'] == first['note']


def test_teacher_source_accepts_actual_runtime_path_string():
    from scripts.coreweave.teacher_incremental_export import _teacher_source_binding
    source = json.loads(_fixture()[3]['teacher-source.json'])
    source['actual_task_runtime'] = '/work/datajobs/runtime/ts-host'
    cases = _fixture()[3]['source-cases.jsonl']
    assert _teacher_source_binding(json.dumps(source).encode(), batch=19,
        source_sha256=hashlib.sha256(cases).hexdigest())['actual_task_runtime'] == source['actual_task_runtime']


def test_owner_poller_reloads_active_config_and_checks_teacher_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.coreweave.teacher_incremental_export as module
    context = _fixture()[3]
    source = json.loads(context['teacher-source.json'])
    source['actual_task_runtime'] = '/work/runtime/ts-host'
    teacher_raw = json.dumps(source).encode()
    cases_raw = context['source-cases.jsonl']
    remote = {'/batch/teacher-source.json': teacher_raw, '/batch/source-cases.jsonl': cases_raw,
              '/batch/queue.jsonl': b'{"key":"k"}\n',
              '/batch/journal.jsonl': b'{"event":"start","key":"k"}\n'}
    sandbox = SimpleNamespace(exec=lambda *args, **kwargs: SimpleNamespace(result=lambda **_: SimpleNamespace(
        returncode=0, stdout='boot-actual-1\n', stderr='')))
    monkeypatch.setattr(module, '_cwctl_module', lambda: SimpleNamespace(load_env_file=lambda: None))
    monkeypatch.setattr(module, 'scoped_sdk_session', lambda *_args, **_kwargs:
                        (sandbox, remote.__getitem__))
    seen = []
    config = {'sandbox_id': 'sandbox-1', 'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': {
            'teacher-source.json': '/batch/teacher-source.json',
            'source-cases.jsonl': '/batch/source-cases.jsonl'},
        'directory': str(tmp_path/'exports'), 'role': 'bulk', 'chunk_size': 32,
        'exporter_path': '/owner/export.py', 'flush_partial': False, 'idle_flush_after_seconds': 0,
        'identity': {'sandbox_boot_id': 'boot-actual-1',
            'teacher_source_sha256': hashlib.sha256(teacher_raw).hexdigest(),
            'source_cases_sha256': hashlib.sha256(cases_raw).hexdigest(),
            'actor_sha256': 'a'*64, 'execution_identity_sha256': 'b'*64,
            'teacher_config_sha256': 'f'*64, 'runtime_path': '/work/runtime/ts-host'}}
    config_path = tmp_path/'active.json'
    config_path.write_text(json.dumps(config))
    def poll(**kwargs):
        seen.append((kwargs['batch'], kwargs['flush_partial'], kwargs['journal_snapshot']))
        if len(seen) == 1:
            config['flush_partial'] = True
            config_path.write_text(json.dumps(config))
        return {'state': {'completed_keys': []}, 'training_admission': False, 'exposure': False}
    monkeypatch.setattr(module, 'export_finished_tranches', poll)
    result = module.run_incremental_service(config_path, poll_seconds=0, max_polls=2,
                                             allowed_spool_root=tmp_path)
    assert seen == [(19, True, remote['/batch/journal.jsonl'])] * 2
    assert result['config_changed'] is True
    assert result['identity']['sandbox_boot_id'] == 'boot-actual-1'
    assert json.loads((tmp_path/'exports/service-state.json').read_text())['training_admission'] is False


def test_owner_poller_fails_closed_on_restarted_sandbox_boot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.coreweave.teacher_incremental_export as module
    context = _fixture()[3]
    source = json.loads(context['teacher-source.json'])
    source['actual_task_runtime'] = '/work/runtime/ts-host'
    teacher_raw = json.dumps(source).encode()
    cases_raw = context['source-cases.jsonl']
    remote = {'/batch/teacher-source.json': teacher_raw, '/batch/source-cases.jsonl': cases_raw,
              '/batch/queue.jsonl': b'{"key":"k"}\n',
              '/batch/journal.jsonl': b'{"event":"start","key":"k"}\n'}
    sandbox = SimpleNamespace(exec=lambda *args, **kwargs: SimpleNamespace(result=lambda **_: SimpleNamespace(
        returncode=0, stdout='different-boot\n', stderr='')))
    monkeypatch.setattr(module, '_cwctl_module', lambda: SimpleNamespace(load_env_file=lambda: None))
    monkeypatch.setattr(module, 'scoped_sdk_session', lambda *_args, **_kwargs:
                        (sandbox, remote.__getitem__))
    config = {'sandbox_id': 'sandbox-1', 'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': {
            'teacher-source.json': '/batch/teacher-source.json',
            'source-cases.jsonl': '/batch/source-cases.jsonl'},
        'directory': str(tmp_path/'exports'), 'role': 'bulk', 'chunk_size': 32,
        'exporter_path': '/owner/export.py', 'identity': {
            'sandbox_boot_id': 'expected-boot',
            'teacher_source_sha256': hashlib.sha256(teacher_raw).hexdigest(),
            'source_cases_sha256': hashlib.sha256(cases_raw).hexdigest(),
            'actor_sha256': 'a'*64, 'execution_identity_sha256': 'b'*64,
            'teacher_config_sha256': 'f'*64, 'runtime_path': '/work/runtime/ts-host'}}
    config_path = tmp_path/'active.json'; config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, 'export_finished_tranches', lambda **_: pytest.fail('export must not run'))
    with pytest.raises(ValueError, match='boot identity differs'):
        module.run_incremental_service(config_path, poll_seconds=0, max_polls=1,
                                       allowed_spool_root=tmp_path)
    state = json.loads((tmp_path/'exports/service-state.json').read_text())
    assert 'boot identity differs' in state['identity_error']


@pytest.mark.parametrize('mutation', ['foreign-key', 'duplicate-finish', 'unsafe-key'])
def test_completed_frontier_rejects_forged_or_ambiguous_identity(tmp_path, mutation):
    entries, queue, journal, context, _, finish = _fixture()
    if mutation == 'foreign-key':
        journal = json.dumps(finish('foreign')).encode() + b'\n'
    elif mutation == 'duplicate-finish':
        event = finish(entries[0]['key'])
        conflicting = json.loads(json.dumps(event)); conflicting['status'] = 'rewritten'
        journal = json.dumps(event).encode() + b'\n' + json.dumps(conflicting).encode() + b'\n'
    else:
        row = dict(entries[0], key='../escape')
        queue = json.dumps(row).encode() + b'\n'
        event = finish('../escape'); event['output_accounting']['output_sha256'] = 'a'*64
        journal = json.dumps(event).encode() + b'\n'
        entries = [row]
    if mutation == 'unsafe-key':
        selected = completed_entries(queue, journal)
        outputs = {'../escape': (b'{}\n', b'{}')}
        with pytest.raises(ValueError, match='safe'):
            seal_tranche(batch=19, number=0, entries=selected,
                context={**context, 'queue.jsonl': queue},
                outputs=outputs, directory=tmp_path/'unsafe-tranche-test')
    else:
        with pytest.raises(ValueError):
            completed_entries(queue, journal)


def test_seal_binds_exact_raw_outputs_context_and_private_verification(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    tranche = seal_tranche(batch=19, number=0, entries=selected, context=context,
                           outputs=outputs, directory=tmp_path/'tranche-0000')
    assert tranche['keys'] == [row['key'] for row in entries]
    assert tranche['training_admission'] is False and tranche['exposure'] is False
    assert tranche['archive_proof']['status'] == 'passed'
    assert seal_tranche(batch=19, number=0, entries=selected, context=context,
                        outputs=outputs, directory=tmp_path/'tranche-0000') == tranche
    receipt = {'private': True, 'run': 'werg/natlang/test', 'name': 'batch/tranche.tar.zst',
        'identity': {'bytes': tranche['archive_receipt']['archive_bytes'],
                     'sha256': tranche['archive_receipt']['archive_sha256']},
        'independent_verification': {'bytes': tranche['archive_receipt']['archive_bytes'],
                     'sha256': tranche['archive_receipt']['archive_sha256']}, 'verified_at': 123.0}
    identity = receipt['identity']
    receipt_path = tmp_path/'private.json'; receipt_path.write_text(json.dumps(receipt))
    accepted = accept_private_receipt(receipt_path, tranche)
    assert accepted['private_export']['identity'] == receipt['identity']
    receipt['independent_verification']['sha256'] = '0'*64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='independent full-object'):
        accept_private_receipt(receipt_path, tranche)
    receipt['independent_verification'] = {
        'status': 'awaiting_target_readback', 'bytes': identity['bytes'], 'sha256': identity['sha256']}
    receipt['identity'] = identity
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='independent full-object'):
        accept_private_receipt(receipt_path, tranche)


def test_seal_rejects_output_manifest_mismatch_without_rewriting_tranche(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    broken = dict(outputs)
    raw, manifest = broken[entries[0]['key']]
    value = json.loads(manifest); value['source'] = '/other/source.jsonl'
    broken[entries[0]['key']] = raw, json.dumps(value).encode()
    with pytest.raises(ValueError, match='does not bind'):
        seal_tranche(batch=19, number=0, entries=selected, context=context,
                     outputs=broken, directory=tmp_path/'tranche')


def test_retry_with_changed_tranche_bytes_does_not_mutate_sealed_payload(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    directory = tmp_path/'tranche'
    seal_tranche(batch=19, number=0, entries=selected, context=context,
                 outputs=outputs, directory=directory)
    result_path = directory/'outputs/0000-s7-task-000-00.results.jsonl'
    before = result_path.read_bytes()
    changed = dict(outputs)
    changed_payload = b'{"changed":true}\n'
    changed_manifest = json.loads(outputs[entries[0]['key']][1])
    changed_sha = hashlib.sha256(changed_payload).hexdigest()
    changed_manifest['output_sha256'] = changed_sha
    changed[entries[0]['key']] = (changed_payload, json.dumps(changed_manifest).encode())
    changed_entries = json.loads(json.dumps(selected))
    changed_entries[0]['finish_event']['output_accounting']['output_sha256'] = changed_sha
    with pytest.raises(ValueError, match='identity or payload bindings differ'):
        seal_tranche(batch=19, number=0, entries=changed_entries, context=context,
                     outputs=changed, directory=directory)
    assert result_path.read_bytes() == before


def test_spool_prune_requires_private_full_get_and_retains_member_manifest(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    directory = tmp_path/'tranche'
    tranche = seal_tranche(batch=19, number=0, entries=selected, context=context,
        outputs=outputs, directory=directory)
    archive = Path(tranche['archive_receipt']['archive'])
    raw_member = directory/'outputs/0000-s7-task-000-00.results.jsonl'
    receipt_path = tmp_path/'private.json'
    receipt_path.write_text(json.dumps({'private': True, 'run': 'werg/natlang/test',
        'name': 'tranche.tar.zst', 'identity': {'bytes': archive.stat().st_size,
            'sha256': tranche['archive_receipt']['archive_sha256']},
        'independent_verification': {'bytes': archive.stat().st_size,
            'sha256': tranche['archive_receipt']['archive_sha256']}}))
    with pytest.raises(ValueError, match='private receipt binding'):
        prune_verified_local_tranche(tranche)
    assert archive.exists() and raw_member.exists()
    accepted = accept_private_receipt(receipt_path, tranche)
    pruning = prune_verified_local_tranche(accepted)
    assert pruning['pruned'] is True and pruning['archive_bytes_removed'] > 0
    assert not archive.exists() and not raw_member.exists()
    retained = json.loads((directory/'tranche-0000.json').read_text())
    assert retained['archive_receipt']['manifest']['files']
    assert (tmp_path/'teacher-batch-0019-tranche-0000.receipt.json').is_file()


def test_tranche_receipt_resumes_after_archive_was_committed(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    selected = completed_entries(queue, journal)
    directory = tmp_path/'tranche'
    sealed = seal_tranche(batch=19, number=0, entries=selected, context=context,
                          outputs=outputs, directory=directory)
    (directory/'tranche-0000.json').unlink()
    resumed = seal_tranche(batch=19, number=0, entries=selected, context=context,
                           outputs=outputs, directory=directory)
    assert resumed['archive_receipt']['archive_sha256'] == sealed['archive_receipt']['archive_sha256']
    assert json.loads((directory/'tranche-0000.json').read_text())['keys'] == sealed['keys']


def test_owner_exporter_checkpoints_chunks_and_resumes_sealed_tranche(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    context_paths = {'teacher-source.json': '/batch/teacher-source.json',
                     'source-cases.jsonl': '/batch/source-cases.jsonl',
                     'repair-queue.jsonl': '/control/repair-queue.jsonl'}
    remote = {'/batch/queue.jsonl': queue, '/batch/journal.jsonl': journal,
              '/batch/teacher-source.json': context['teacher-source.json'],
              '/batch/source-cases.jsonl': context['source-cases.jsonl'],
              '/control/repair-queue.jsonl': context['repair-queue.jsonl']}
    for row in entries:
        remote[row['output']] = outputs[row['key']][0]
        remote[row['output']+'.manifest.json'] = outputs[row['key']][1]
    calls = []
    def export(tranche, receipt_path):
        calls.append(tranche['keys'])
        ident = {'bytes': tranche['archive_receipt']['archive_bytes'],
                 'sha256': tranche['archive_receipt']['archive_sha256']}
        if len(calls) == 1:
            raise RuntimeError('simulate owner process interruption after sealing')
        return {**tranche, 'private_export': {'identity': ident}}
    kwargs = {'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': context_paths,
        'directory': tmp_path/'exports', 'read_file': remote.__getitem__,
        'export_file': export, 'chunk_size': 2}
    with pytest.raises(RuntimeError, match='interruption'):
        export_finished_tranches(**kwargs)
    sealed = json.loads((tmp_path/'exports/tranche-0000/tranche-0000.json').read_text())
    assert sealed['keys'] == [row['key'] for row in entries]
    assert sealed['private_export'] is None
    result = export_finished_tranches(**kwargs)
    assert result['durable_completed_keys'] == 2
    assert calls[0] == calls[1] == [row['key'] for row in entries]
    assert export_finished_tranches(**kwargs)['durable_completed_keys'] == 2
    with pytest.raises(ValueError, match='different batch or queue'):
        export_finished_tranches(**{**kwargs, 'batch': 20})


def test_verified_private_receipt_survives_crash_before_frontier_advance(tmp_path):
    entries, queue, journal, context, outputs, _ = _fixture()
    context_paths = {'teacher-source.json': '/batch/teacher-source.json',
                     'source-cases.jsonl': '/batch/source-cases.jsonl'}
    remote = {'/batch/queue.jsonl': queue, '/batch/journal.jsonl': journal,
              '/batch/teacher-source.json': context['teacher-source.json'],
              '/batch/source-cases.jsonl': context['source-cases.jsonl']}
    for row in entries:
        remote[row['output']] = outputs[row['key']][0]
        remote[row['output']+'.manifest.json'] = outputs[row['key']][1]
    calls = []
    def export(tranche, receipt_path):
        calls.append(tranche['keys'])
        identity = {'bytes': tranche['archive_receipt']['archive_bytes'],
                    'sha256': tranche['archive_receipt']['archive_sha256']}
        receipt = {'private': True, 'run': 'werg/natlang/test', 'name': 'tranche.tar.zst',
            'identity': identity, 'independent_verification': identity, 'verified_at': 124.0}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt))
        accept_private_receipt(receipt_path, tranche)
        raise RuntimeError('crash after private verification, before frontier commit')
    kwargs = {'batch': 19, 'queue_path': '/batch/queue.jsonl',
        'journal_path': '/batch/journal.jsonl', 'context_paths': context_paths,
        'directory': tmp_path/'exports', 'read_file': remote.__getitem__,
        'export_file': export, 'chunk_size': 2, 'prune_after_private_verification': True}
    with pytest.raises(RuntimeError, match='before frontier commit'):
        export_finished_tranches(**kwargs)
    result = export_finished_tranches(**{**kwargs,
        'export_file': lambda *_: pytest.fail('already verified private object must not upload again')})
    assert result['durable_completed_keys'] == 2
    assert len(calls) == 1
    tranche_dir = tmp_path/'exports/tranche-0000'
    durable_manifest = json.loads((tranche_dir/'tranche-0000.json').read_text())
    assert not Path(durable_manifest['archive_receipt']['archive']).exists()
    assert sorted(path.name for path in tranche_dir.iterdir()) == ['tranche-0000.json']


def test_final_boundary_requires_exact_partition_and_all_raw_control_files(tmp_path):
    from scripts.coreweave.teacher_generation_guardian import assess
    entries, queue, journal, context, _, finish = _fixture()
    events = [json.loads(line) for line in journal.splitlines()]
    finishes = [row for row in events if row.get('event') == 'finish']
    score = {'schema': 'natlang.s7-task-rollout-score/1', 'rollouts': 2,
             'outcome_counts': {'task_success': 2}}
    completion = assess(entries, finishes, [0], score)
    files = {**context, 'journal.jsonl': journal, 'score.json': json.dumps(score).encode(),
             'rewards.jsonl': b'{}\n', 'completion.json': json.dumps(completion).encode(),
             'repair-queue.jsonl': context['repair-queue.jsonl']}
    completion.update(batch=19, bindings={name: {'bytes': len(files[name]),
        'sha256': hashlib.sha256(files[name]).hexdigest()} for name in (
            'queue.jsonl', 'journal.jsonl', 'score.json', 'teacher-source.json')})
    files['completion.json'] = json.dumps(completion).encode()
    keys = [row['key'] for row in entries]
    with pytest.raises(ValueError, match='do not partition'):
        seal_boundary(batch=19, queue_raw=queue, exported_keys=keys[:1], repair_keys=[],
            unstarted_keys=[], files=files, directory=tmp_path/'bad')
    boundary = seal_boundary(batch=19, queue_raw=queue, exported_keys=keys,
        repair_keys=[], unstarted_keys=[], files=files, directory=tmp_path/'final')
    assert boundary['archive_proof']['status'] == 'passed'
    assert boundary['boundary']['training_admission'] is False
    resumed = seal_boundary(batch=19, queue_raw=queue, exported_keys=keys,
        repair_keys=[], unstarted_keys=[], files=files, directory=tmp_path/'final')
    assert resumed['archive_receipt']['archive_sha256'] == boundary['archive_receipt']['archive_sha256']
    altered = dict(files, **{'score.json': b'{"changed":true}'})
    with pytest.raises(ValueError, match='does not bind current score'):
        seal_boundary(batch=19, queue_raw=queue, exported_keys=keys,
            repair_keys=[], unstarted_keys=[], files=altered, directory=tmp_path/'final')


def test_final_boundary_preserves_actual_repair_error_without_complete_batch_claim(tmp_path):
    from scripts.coreweave.teacher_generation_guardian import assess
    entries, queue, journal, context, _, finish = _fixture()
    row = entries[0]
    error = {'index': 0, 'program_id': row['task_id'], 'program_ir_sha256': '1'*64,
             'source_sha256': hashlib.sha256(context['source-cases.jsonl']).hexdigest(),
             'error': 'provider unavailable'}
    error_raw = json.dumps(error).encode()
    failed_accounting = {'version': 'natlang.supervisor_output_accounting/2', 'complete': False,
        'manifest_valid': False, 'manifest_errors': ['terminal_error'], 'expected_jobs': 1,
        'job_states': [{'index': 0, 'program_id': row['task_id'], 'digest': '1'*64,
                        'state': 'failed', 'terminal_error': {'path': '/batch/jobs/000000.error.json',
                            'sha256': hashlib.sha256(error_raw).hexdigest()}}]}
    failed = {'event': 'finish', 'key': row['key'], 'output_accounting': failed_accounting}
    outcomes = [dict(entry, finish=failed if entry['key'] == row['key'] else None) for entry in entries]
    # The shared assessor is authoritative for repair disposition; its exact
    # schema contains no synthetic batch/can_advance fields at top level.
    score = {'schema': 'natlang.s7-task-rollout-score/1', 'rollouts': 2,
             'outcome_counts': {'task_success': 1, 'provider_failure': 1}}
    completion = assess(entries, [failed], [2], score)
    journal_bytes = b''.join(json.dumps(item).encode()+b'\n' for item in [
        {'event': 'start', 'key': row['key']}, failed])
    completion.update(batch=19, individual_outcomes=outcomes, bindings={
        'queue.jsonl': {'bytes': len(queue), 'sha256': hashlib.sha256(queue).hexdigest()},
        'journal.jsonl': {'bytes': len(journal_bytes), 'sha256': hashlib.sha256(journal_bytes).hexdigest()},
        'score.json': {'bytes': len(json.dumps(score).encode()),
                       'sha256': hashlib.sha256(json.dumps(score).encode()).hexdigest()},
        'teacher-source.json': {'bytes': len(context['teacher-source.json']),
            'sha256': hashlib.sha256(context['teacher-source.json']).hexdigest()}})
    files = {**context, 'queue.jsonl': queue, 'journal.jsonl': journal_bytes,
        'score.json': json.dumps(score).encode(), 'rewards.jsonl': b'{}\n',
        'completion.json': json.dumps(completion).encode(), 'repair-queue.jsonl': b'{}\n',
        f"repair-payloads/{row['key']}/000000.error.json": error_raw}
    boundary = seal_boundary(batch=19, queue_raw=queue, exported_keys=[],
        repair_keys=[row['key']], unstarted_keys=[entries[1]['key']], files=files,
        directory=tmp_path/'repair-boundary')
    assert boundary['boundary']['batch_can_advance'] is False
    assert boundary['boundary']['training_admission'] is False
