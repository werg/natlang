import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import continuous_storage_manager as manager


def _fixture(tmp_path, *, free=10**9, owner_fragment=None, external_uuid='fixture-uuid'):
    checkpoint = tmp_path / 'internal' / 'cycle' / 'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'full-state')
    owner = tmp_path / 'owner.json'
    owner.write_text(json.dumps({'pid': os.getpid(), 'unit': 'fixture', 'runtime': '/frozen/runtime'}))
    cursor = tmp_path / 'cursor.json'
    cursor.write_text(json.dumps({'checkpoint': {'path': str(checkpoint), 'sha256': 'source-bound'},
                                 'cycle': 9, 'phase': 'supervised', 'stage_index': 1}))
    config = tmp_path / 'config.json'
    config.write_text('{}')
    external = tmp_path / 'external'
    external.mkdir()
    policy = {'_path': str(tmp_path / 'policy.json'), 'status_path': str(tmp_path / 'status.json'),
              'journal_path': str(tmp_path / 'events.jsonl'), 'internal_path': str(tmp_path / 'internal'),
              'owner_path': str(owner), 'cursor_path': str(cursor),
              'config_path': str(config),
              'owner_command_contains': owner_fragment or Path(sys.executable).name,
              'retention_script': str(tmp_path/'retention.py'),
              'retention_plan_path': str(tmp_path/'retention-plan.json'),
              'project_python': sys.executable,
              'archive_root': str(tmp_path/'internal'),
              'internal_min_free_bytes': 1, 'checkpoint_peak_factor': 2,
              'external_mount': str(external), 'external_uuid': external_uuid,
              'external_floor_bytes': 1, 'archive_script': '/archive.py',
              'archive_root': str(checkpoint.parent), 'mellum_local_pointer': str(tmp_path / 'missing-local.json'),
              'mellum_private_pointer': str(tmp_path / 'missing-private.json'), 'mellum_protected_paths': []}
    usage = type('Usage', (), {'total': 10**9, 'used': 10**9-free, 'free': free})()
    return policy, usage, checkpoint


def test_manager_reports_bound_owner_and_checkpoint_and_durable_status(tmp_path, monkeypatch):
    policy, usage, _ = _fixture(tmp_path)
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    status = manager.run_once(policy)
    assert status['state'] == 'healthy'
    assert status['owner']['unit'] == 'fixture'
    assert status['checkpoint']['cycle'] == 9
    assert status['next_write_admission']['mode'] == 'monitor-and-recover'
    assert status['next_write_admission']['live_writer_guard_installed'] is False
    assert status['next_write_admission']['estimated_required_headroom_bytes'] > 0
    assert json.loads(Path(policy['status_path']).read_text())['checkpoint'] == status['checkpoint']
    assert len(Path(policy['journal_path']).read_text().splitlines()) == 1


def test_manager_rejects_owner_command_change_and_selected_path_escape(tmp_path, monkeypatch):
    policy, usage, checkpoint = _fixture(tmp_path, owner_fragment='definitely-not-the-live-command')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    status = manager.inspect(policy)
    assert 'command no longer matches' in status['internal_state_error']
    cursor = json.loads(Path(policy['cursor_path']).read_text())
    cursor['checkpoint']['path'] = str(tmp_path / 'outside.pt')
    Path(policy['cursor_path']).write_text(json.dumps(cursor))
    policy['owner_command_contains'] = Path(sys.executable).name
    status = manager.inspect(policy)
    assert 'escapes' in status['internal_state_error']
    assert checkpoint.exists()


def test_invalid_owner_or_cursor_never_reaches_archive_planner(tmp_path, monkeypatch):
    policy, usage, _ = _fixture(tmp_path, free=2)
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    calls = []
    monkeypatch.setattr(manager, '_archive_dry_run', lambda *args: calls.append(args))
    cursor = json.loads(Path(policy['cursor_path']).read_text())
    cursor['checkpoint']['path'] = str(tmp_path/'escape.pt')
    Path(policy['cursor_path']).write_text(json.dumps(cursor))
    status = manager.inspect(policy)
    assert 'escapes' in status['internal_state_error']
    assert not calls
    assert status['action_required'] == 'Owner or selected checkpoint could not be verified.'


def test_pressure_only_dry_runs_bounded_archive_after_external_mount_check(tmp_path, monkeypatch):
    policy, usage, _ = _fixture(tmp_path, free=2)
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    calls = []
    monkeypatch.setattr(manager, '_archive_dry_run', lambda p, b, paths=(): calls.append((p, b, paths)) or {'returncode': 0})
    status = manager.inspect(policy)
    assert status['state'] == 'attention'
    assert len(calls) == 1
    assert calls[0][1] > 0
    assert calls[0][2] and status['archive_protected_paths'] == calls[0][2]
    policy['external_uuid'] = 'different-drive'
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    status = manager.inspect(policy)
    assert status['external']['identity_ok'] is False
    assert status['state'] == 'attention'
    assert len(calls) == 1
    assert 'External mount identity is not verified' in status['action_required']


def test_mellum_plan_mode_never_applies_lfm_archive_even_when_policy_allows_it(tmp_path, monkeypatch):
    policy, usage, _ = _fixture(tmp_path, free=2)
    policy['archive_apply_enabled'] = True
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    monkeypatch.setattr(manager, '_archive_dry_run', lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('Mellum plan command must not run unrelated LFM archive planning')))
    monkeypatch.setattr(manager, '_archive_apply', lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('Mellum plan command must not apply unrelated LFM archive')))
    status = manager.inspect(policy, mellum_prune_plan=True)
    assert 'skipped unrelated LFM archive planning or apply' in status['action_required']
    assert 'archive_dry_run' not in status
    assert 'archive_apply' not in status


def test_existing_lfm_plan_is_bound_for_review_and_never_applied(tmp_path):
    policy, _, _ = _fixture(tmp_path)
    plan_path = Path(policy['retention_plan_path'])
    plan_path.write_text(json.dumps({'recovery': {'checkpoint': {'sha256': 'old-head'}},
        'owner_identity': {'pid': 10, 'unit': 'old-owner', 'runtime': '/old'},
        'candidates': [{'bytes': 1234, 'path': '/closed/one.pt'}]}))
    owner = {'pid': os.getpid(), 'unit': 'live-owner', 'runtime': '/frozen/runtime'}
    value = manager._lfm_retention_reference(policy, 'new-head', owner)
    assert value['candidate_count'] == 1 and value['candidate_bytes'] == 1234
    assert value['current_binding_matches'] is False
    assert value['review_required'] is True and value['applied'] is False


def test_archive_command_is_cycle_scoped_floor_bound_and_preview_only(tmp_path, monkeypatch):
    policy, _, _ = _fixture(tmp_path)
    policy.update(project_python='/project/python', archive_script='/repo/scripts/archive_to_external.py',
                  archive_root='/home/data/lfm/rl/cycles', external_floor_bytes=80*1024**3)
    seen = []
    monkeypatch.setattr(manager.subprocess, 'run', lambda command, **kwargs:
        seen.append((command, kwargs)) or type('Result', (), {'returncode': 0, 'stdout': '{}', 'stderr': ''})())
    result = manager._archive_dry_run(policy, 3*1024**3, ['/active/cycle', '/active/cycle/stage'])
    command = result['command']
    assert command[:2] == ['/project/python', '/repo/scripts/archive_to_external.py']
    assert command[command.index('--root')+1] == '/home/data/lfm/rl/cycles'
    assert command[command.index('--min-age-hours')+1] == '0'
    assert command[command.index('--idle-run-min-age-hours')+1] == '0'
    assert command.count('--exclude') == 2
    assert command[command.index('--hdd-floor-gb')+1] == '80.0'
    assert '--apply' not in command
    assert seen[0][1]['timeout'] == 1800


def test_archive_command_caps_batches_and_reserves_pending_mellum_copy(tmp_path):
    policy, _, _ = _fixture(tmp_path)
    pointer = tmp_path/'mellum-private.json'
    pointer.write_text(json.dumps({'local_copy_pending': True,
        'checkpoint': {'bytes': 48*1024**3, 'sha256': 'a'*64}}))
    policy.update(project_python='/project/python', archive_script='/archive.py',
                  external_floor_bytes=80*1024**3, external_archive_safety_bytes=1*1024**3,
                  archive_batch_max_bytes=16*1024**3, mellum_private_pointer=str(pointer))
    command = manager._archive_command(policy, 40*1024**3)
    assert command[command.index('--budget-gb')+1] == '16.0'
    assert command[command.index('--hdd-floor-gb')+1] == '129.0'
    pointer.write_text(json.dumps({'local_copy_pending': True,
        'checkpoint': {'bytes': 0, 'sha256': 'a'*64}}))
    with pytest.raises(ValueError, match='exact byte size'):
        manager._archive_command(policy, 40*1024**3)


def test_archive_apply_is_opt_in_and_revalidates_current_owner_head_and_mount(tmp_path, monkeypatch):
    policy, usage, _ = _fixture(tmp_path, free=2)
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager, '_archive_dry_run', lambda *args: {'returncode': 0, 'stdout': '{}'})
    calls = []
    monkeypatch.setattr(manager, '_archive_apply', lambda *args: calls.append(args) or {
        'executed': True, 'returncode': 0, 'moved_files': 1, 'moved_bytes': 123})
    status = manager.inspect(policy)
    assert not calls
    assert not status['actions_applied']
    policy['archive_apply_enabled'] = True
    status = manager.inspect(policy)
    assert len(calls) == 1
    assert calls[0][0] is policy and calls[0][1] > 0
    assert calls[0][3]['owner']['pid'] == os.getpid()
    assert calls[0][3]['checkpoint']['path'] == status['checkpoint']['path']
    assert status['actions_applied'] == ['verified-closed-file-archive']
    assert status['archive_apply']['moved_bytes'] == 123


def test_archive_apply_command_is_explicit_and_keeps_exact_exclusions(tmp_path):
    policy, _, _ = _fixture(tmp_path)
    policy.update(project_python='/project/python', archive_script='/repo/scripts/archive_to_external.py',
                  archive_root='/home/data/lfm/rl/cycles', archive_destination='/storage/archive',
                  external_floor_bytes=80*1024**3)
    command = manager._archive_command(policy, 3*1024**3, ['/active/cycle'], apply=True)
    assert '--apply' in command
    assert command[command.index('--archive')+1] == '/storage/archive'
    assert command[command.index('--exclude')+1] == '/active/cycle'
    assert command[command.index('--root')+1] == '/home/data/lfm/rl/cycles'


def test_archive_apply_revalidates_external_identity_and_floor(tmp_path, monkeypatch):
    policy, _, checkpoint = _fixture(tmp_path)
    owner = manager._owner_status(policy['owner_path'], policy['owner_command_contains'])
    current = manager._checkpoint_status(policy['cursor_path'], policy['internal_path'])
    expected = {'owner': {k: owner.get(k) for k in
                          ('pid', 'unit', 'runtime', 'scope', 'command_sha256')},
                'config_sha256': manager._stable_file_sha256(policy['config_path']),
                'checkpoint': {k: current.get(k) for k in (
                    'path', 'device', 'inode', 'bytes', 'mtime_ns', 'cursor_checkpoint_sha256',
                    'cycle', 'phase', 'stage_index')}}
    policy['external_mount'] = str(tmp_path/'external')
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 0})())
    result = manager._archive_apply(policy, 1024, (), expected)
    assert result['applied'] is False
    assert 'floor' in result['skipped']
    checkpoint.write_bytes(b'changed')
    result = manager._archive_apply(policy, 1024, (), expected)
    assert result['applied'] is False
    assert 'changed after dry-run' in result['skipped']


def test_archive_apply_runs_only_after_exact_revalidation(tmp_path, monkeypatch):
    policy, _, checkpoint = _fixture(tmp_path)
    policy['external_mount'] = str(tmp_path/'external')
    owner = manager._owner_status(policy['owner_path'], policy['owner_command_contains'])
    current = manager._checkpoint_status(policy['cursor_path'], policy['internal_path'])
    expected = {'owner': {k: owner.get(k) for k in
                          ('pid', 'unit', 'runtime', 'scope', 'command_sha256')},
                'config_sha256': manager._stable_file_sha256(policy['config_path']),
                'checkpoint': {k: current.get(k) for k in (
                    'path', 'device', 'inode', 'bytes', 'mtime_ns', 'cursor_checkpoint_sha256',
                    'cycle', 'phase', 'stage_index')}}
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 100})())
    calls = []
    monkeypatch.setattr(manager.subprocess, 'run', lambda command, **kwargs:
        calls.append((command, kwargs)) or type('Result', (), {
            'returncode': 0, 'stdout': '{"moved_files": 1, "moved_gb": 0.5}\n', 'stderr': ''})())
    result = manager._archive_apply(policy, 2048, ['/active/cycle'], expected)
    assert result['executed'] is True
    assert result['returncode'] == 0 and result['moved_files'] == 1
    assert result['moved_bytes'] == int(0.5*manager.GIB)
    assert '--apply' in calls[0][0]
    assert calls[0][0][calls[0][0].index('--exclude')+1] == '/active/cycle'
    assert calls[0][1]['timeout'] == 1800


def test_archive_protection_covers_active_and_previous_cycles_and_config_refs(tmp_path):
    policy, _, checkpoint = _fixture(tmp_path)
    cycle_root = Path(policy['internal_path'])/'cycles'
    active_cycle = cycle_root/'cycle-000028'
    active_checkpoint = active_cycle/'stage'/'learner'/'checkpoint.pt'
    active_checkpoint.parent.mkdir(parents=True)
    active_checkpoint.write_bytes(b'active')
    config_source = cycle_root/'cycle-000021'/'parent'/'checkpoint.pt'
    config_source.parent.mkdir(parents=True)
    config_source.write_bytes(b'parent')
    policy['archive_root'] = str(cycle_root)
    policy['config_path'] = str(tmp_path/'config.json')
    Path(policy['config_path']).write_text(json.dumps({'continue_from': str(config_source)}))
    paths = manager._archive_protected_paths(policy, {'path': str(active_checkpoint)})
    assert str(active_cycle) in paths
    assert str(cycle_root/'cycle-000027') in paths
    assert str(config_source.parent) in paths


def test_mellum_current_local_and_private_are_pinned_and_never_pruned(tmp_path, monkeypatch):
    local = tmp_path / 'mellum-local.pt'
    local.write_bytes(b'current-local')
    previous = tmp_path / 'mellum-previous.pt'
    previous.write_bytes(b'previous-local')
    local_pointer = tmp_path / 'local.json'
    local_pointer.write_text(json.dumps({'global_step': 36504, 'checkpoint': {
        'path': str(local), 'sha256': 'a'*64, 'bytes': local.stat().st_size}}))
    private_pointer = tmp_path / 'private.json'
    private_pointer.write_text(json.dumps({'global_step': 36596, 'local_copy_pending': True,
        'checkpoint': {'path': '/remote/mellum.pt', 'sha256': 'b'*64, 'bytes': 123},
        'receipt': str(tmp_path / 'not-yet-verified.json')}))
    policy, usage, _ = _fixture(tmp_path)
    policy['mellum_local_pointer'] = str(local_pointer)
    policy['mellum_private_pointer'] = str(private_pointer)
    policy['mellum_protected_paths'] = [str(previous)]
    cache = tmp_path/'mellum-cache'
    cache.mkdir()
    old_digest = 'c'*64
    old_pin = cache/f'mellum-{old_digest}.pt'
    old_pin.write_bytes(b'old-full-pin')
    policy['mellum_cache_root'] = str(cache)
    policy['mellum_receipt_root'] = str(tmp_path)
    monkeypatch.setattr(manager, '_mount_uuid', lambda _: 'fixture-uuid')
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    status = manager.inspect(policy)
    assert str(local) in status['mellum']['protected_paths']
    assert str(previous) in status['mellum']['protected_paths']
    assert status['mellum']['private']['local_copy_pending'] is True
    assert status['mellum']['private']['private_receipt_valid'] is False
    assert status['mellum']['automatic_pruning'] is False
    assert len(status['mellum']['cache_inventory']) == 1
    assert status['mellum']['cache_inventory'][0]['deletion_eligible'] is False
    assert local.exists() and previous.exists()


def test_missing_selected_checkpoint_is_actionable_not_healthy(tmp_path, monkeypatch):
    policy, usage, checkpoint = _fixture(tmp_path)
    checkpoint.unlink()
    monkeypatch.setattr(manager.shutil, 'disk_usage', lambda _: usage)
    status = manager.inspect(policy)
    assert status['state'] == 'attention'
    assert 'No such file' in status['internal_state_error']


def _mellum_prune_fixture(tmp_path):
    external = tmp_path/'external'
    cache, receipts = external/'mellum-cache', tmp_path/'receipts'
    external.mkdir()
    cache.mkdir(); receipts.mkdir()
    objects = {}
    for name, body in [('current', b'current full state'), ('previous', b'previous full state'),
                       ('stale', b'stale full state')]:
        digest = __import__('hashlib').sha256(body).hexdigest()
        path = cache/f'mellum-{digest}.pt'
        path.write_bytes(body)
        identity = {'sha256': digest, 'bytes': len(body)}
        receipt = {'role': 'mellum', 'source': f'/work/resume/independent-exports/mellum-{digest}.pt',
            'name': f'mellum/{digest}/mellum-{digest}.pt', 'run': 'werg/natlang/test-run',
            'private': True, 'identity': identity, 'independent_verification': dict(identity)}
        receipt_path = receipts/f'{digest}.private-receipt.json'
        receipt_path.write_text(json.dumps(receipt))
        objects[name] = {'path': path, 'receipt': receipt_path, 'receipt_value': receipt}
    latest_body = b'latest private export'
    latest_sha = __import__('hashlib').sha256(latest_body).hexdigest()
    latest_id = {'sha256': latest_sha, 'bytes': len(latest_body)}
    latest_receipt = {'role': 'mellum', 'source': f'/work/resume/independent-exports/mellum-{latest_sha}.pt',
        'name': f'mellum/{latest_sha}/mellum-{latest_sha}.pt', 'run': 'werg/natlang/test-run',
        'private': True, 'identity': latest_id, 'independent_verification': dict(latest_id)}
    latest_receipt_path = receipts/f'{latest_sha}.private-receipt.json'
    latest_receipt_path.write_text(json.dumps(latest_receipt))
    current, previous, stale = (objects[key] for key in ('current', 'previous', 'stale'))
    local_pointer = tmp_path/'local-pointer.json'
    local_pointer.write_text(json.dumps({'passed': True, 'global_step': 30,
        'checkpoint': {'path': str(current['path']), 'sha256': current['path'].name[7:-3],
                       'bytes': current['path'].stat().st_size}, 'private_receipt': str(current['receipt'])}))
    private_pointer = tmp_path/'private-pointer.json'
    private_pointer.write_text(json.dumps({'global_step': 31,
        'checkpoint': {'path': f'/work/resume/independent-exports/mellum-{latest_sha}.pt',
                       'sha256': latest_sha, 'bytes': latest_id['bytes']}, 'receipt': str(latest_receipt_path)}))
    policy = {'mellum_cache_root': str(cache), 'mellum_receipt_root': str(receipts),
        '_operator_binding': {'script_path': '/reviewed/continuous_storage_manager.py',
            'script_sha256': 'a'*64, 'policy_path': '/reviewed/policy.json',
            'policy_sha256': 'b'*64},
        'external_mount': str(external), 'external_uuid': 'fixture-uuid',
        'mellum_local_pointer': str(local_pointer), 'mellum_private_pointer': str(private_pointer),
        'mellum_protected_paths': [str(previous['path'])],
        'mellum_local_proof_cache': str(tmp_path/'proofs.json'),
        'mellum_retention_state_path': str(tmp_path/'retention-state.json'),
        'mellum_prune_plan_path': str(tmp_path/'prune-plan.json'),
        'mellum_prune_lock_path': str(tmp_path/'prune.lock')}
    return policy, objects, latest_receipt


def _patch_mellum_mount(monkeypatch):
    monkeypatch.setattr(manager, '_mount_uuid', lambda path: 'fixture-uuid')


def _patch_mellum_lineage(monkeypatch, objects):
    model_identity = 'mellum:' + 'c'*64
    current_sha = objects['current']['path'].name[7:-3]
    previous_sha = objects['previous']['path'].name[7:-3]
    stale_sha = objects['stale']['path'].name[7:-3]
    current_ancestors = [previous_sha, stale_sha]
    values = {
        current_sha: {'sha256': current_sha, 'global_step': 30, 'model_identity': model_identity,
                      'native_snapshot_ancestors': current_ancestors,
                      'native_snapshot_ancestors_sha256': __import__('hashlib').sha256(
                          json.dumps(current_ancestors, separators=(',', ':')).encode()).hexdigest(),
                      'checkpoint_stat': {'device': 1, 'inode': 2, 'bytes': 30,
                          'mtime_ns': 3, 'ctime_ns': 4, 'links': 1, 'allocated_bytes': 30},
                      'metadata_reader': {'sha256': 'reader'}},
        previous_sha: {'sha256': previous_sha, 'global_step': 25, 'model_identity': model_identity,
                       'native_snapshot_ancestors': [], 'metadata_reader': {'sha256': 'reader'}},
        stale_sha: {'sha256': stale_sha, 'global_step': 20, 'model_identity': model_identity,
                    'native_snapshot_ancestors': [], 'metadata_reader': {'sha256': 'reader'}},
    }
    monkeypatch.setattr(manager, '_mellum_checkpoint_lineage', lambda policy, path, sha: values[sha])
    return values


def test_mellum_prune_plan_stream_verifies_private_presence_and_resumes_proofs(tmp_path, monkeypatch):
    policy, objects, latest_receipt = _mellum_prune_fixture(tmp_path)
    _patch_mellum_mount(monkeypatch)
    _patch_mellum_lineage(monkeypatch, objects)
    monkeypatch.setattr(manager, '_authenticated_private_presence', lambda policy, receipt: True)
    result = manager._mellum_cache_maintenance(policy, apply=False)
    assert result['state'] == 'planned'
    assert result['candidate_count'] == 1
    assert result['candidate_bytes'] == objects['stale']['path'].stat().st_size
    assert objects['stale']['path'].exists()
    plan = json.loads(Path(policy['mellum_prune_plan_path']).read_text())
    assert plan['operator'] == policy['_operator_binding']
    assert plan['local_recovery_ready'] is True
    assert plan['latest_private_object_present'] is True
    assert [Path(row['path']) for row in plan['candidates']] == [objects['stale']['path']]
    assert len(plan['verified_current_and_previous_bindings']) == 2
    monkeypatch.setattr(manager, '_hash_mellum_pin', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('proof should be cached')))
    again = manager._mellum_cache_maintenance(policy, apply=False)
    assert again['candidate_count'] == 1
    assert objects['stale']['path'].exists()


def test_mellum_prune_only_unlinks_durable_planned_stale_cache_and_counts_final_link(tmp_path, monkeypatch):
    policy, objects, _ = _mellum_prune_fixture(tmp_path)
    _patch_mellum_mount(monkeypatch)
    _patch_mellum_lineage(monkeypatch, objects)
    monkeypatch.setattr(manager, '_authenticated_private_presence', lambda policy, receipt: True)
    result = manager._mellum_cache_maintenance(policy, apply=True)
    assert result['state'] == 'applied'
    assert not objects['stale']['path'].exists()
    assert objects['current']['path'].exists() and objects['previous']['path'].exists()
    assert len(result['deleted']) == 1
    assert result['deleted'][0]['confirmed_absent'] is True
    saved = json.loads(Path(policy['mellum_prune_plan_path']).read_text())
    assert saved['deletions'][0]['sha256'] == result['deleted'][0]['sha256']


def test_mellum_prune_protects_live_inode_and_aborts_on_pointer_advance(tmp_path, monkeypatch):
    policy, objects, _ = _mellum_prune_fixture(tmp_path)
    _patch_mellum_mount(monkeypatch)
    _patch_mellum_lineage(monkeypatch, objects)
    stale_stat = objects['stale']['path'].stat()
    monkeypatch.setattr(manager, '_authenticated_private_presence', lambda policy, receipt: True)
    monkeypatch.setattr(manager, '_live_file_inodes', lambda: {(stale_stat.st_dev, stale_stat.st_ino)})
    protected = manager._mellum_cache_maintenance(policy, apply=False)
    assert protected['candidate_count'] == 0 and objects['stale']['path'].exists()

    monkeypatch.setattr(manager, '_live_file_inodes', lambda: set())
    original = manager._authenticated_private_presence
    calls = 0
    def advance(policy, receipt):
        nonlocal calls
        calls += 1
        if calls == 2:
            local = json.loads(Path(policy['mellum_local_pointer']).read_text())
            local['global_step'] += 1
            Path(policy['mellum_local_pointer']).write_text(json.dumps(local))
        return original(policy, receipt)
    monkeypatch.setattr(manager, '_authenticated_private_presence', advance)
    result = manager._mellum_cache_maintenance(policy, apply=True)
    assert result['state'] == 'skipped'
    assert objects['stale']['path'].exists()


def test_mellum_prune_rejects_nonancestor_even_with_valid_local_and_private_hashes(tmp_path, monkeypatch):
    policy, objects, _ = _mellum_prune_fixture(tmp_path)
    _patch_mellum_mount(monkeypatch)
    values = _patch_mellum_lineage(monkeypatch, objects)
    stale_sha = objects['stale']['path'].name[7:-3]
    current_lineage = values[next(iter(values))]
    current_lineage['native_snapshot_ancestors'].remove(stale_sha)
    current_lineage['native_snapshot_ancestors_sha256'] = __import__('hashlib').sha256(
        json.dumps(current_lineage['native_snapshot_ancestors'], separators=(',', ':')).encode()).hexdigest()
    monkeypatch.setattr(manager, '_authenticated_private_presence', lambda policy, receipt: True)
    result = manager._mellum_cache_maintenance(policy, apply=False)
    assert result['candidate_count'] == 0
    assert objects['stale']['path'].exists()
    plan = json.loads(Path(policy['mellum_prune_plan_path']).read_text())
    stale = next(row for row in plan['inventory'] if row['sha256'] == stale_sha)
    assert stale['state'] == 'ineligible'
    assert 'ancestor' in stale['reason']


def test_mellum_prune_rejects_symlinked_receipt_and_pointer(tmp_path, monkeypatch):
    policy, objects, _ = _mellum_prune_fixture(tmp_path)
    _patch_mellum_mount(monkeypatch)
    local = json.loads(Path(policy['mellum_local_pointer']).read_text())
    alias = Path(policy['mellum_receipt_root'])/'receipt-alias.json'
    alias.symlink_to(objects['current']['receipt'])
    local['private_receipt'] = str(alias)
    Path(policy['mellum_local_pointer']).write_text(json.dumps(local))
    result = manager._mellum_cache_maintenance(policy, apply=False)
    assert result['state'] == 'skipped'
    assert 'receipt' in result['reason']

    alias.unlink()
    pointer = Path(policy['mellum_local_pointer'])
    pointer_alias = pointer.with_name('local-pointer-alias.json')
    pointer_alias.symlink_to(pointer)
    policy['mellum_local_pointer'] = str(pointer_alias)
    result = manager._mellum_cache_maintenance(policy, apply=False)
    assert result['state'] == 'skipped'
    assert 'regular pointer file' in result['reason']


def test_mellum_prune_lock_rejects_symlink_without_touching_target(tmp_path, monkeypatch):
    policy, objects, _ = _mellum_prune_fixture(tmp_path)
    _patch_mellum_mount(monkeypatch)
    victim = tmp_path/'unrelated-lock-target'
    victim.write_bytes(b'untouched')
    alias = tmp_path/'prune.lock'
    alias.symlink_to(victim)
    policy['mellum_prune_lock_path'] = str(alias)
    result = manager._mellum_cache_maintenance(policy, apply=False)
    assert result['state'] == 'skipped'
    assert 'cannot open exact Mellum prune lock' in result['reason']
    assert victim.read_bytes() == b'untouched'


def test_mellum_prune_refuses_changed_external_mount_identity(tmp_path, monkeypatch):
    policy, objects, _ = _mellum_prune_fixture(tmp_path)
    monkeypatch.setattr(manager, '_mount_uuid', lambda path: 'different-uuid')
    result = manager._mellum_cache_maintenance(policy, apply=False)
    assert result['state'] == 'skipped'
    assert 'declared external mount' in result['reason']
    assert all(item['path'].exists() for item in objects.values())


def test_mellum_lineage_reader_is_exactly_bound_and_metadata_only(tmp_path):
    checkpoint = tmp_path/'checkpoint.json'
    value = {'step': 7, 'model_config': {'architecture': 'mellum', 'base_inventory_sha256': 'c'*64},
        'training': {'global_step': 7, 'options': {'native_snapshot_ancestors': ['a'*64, 'b'*64]}}}
    checkpoint.write_text(json.dumps(value))
    reader_source = tmp_path/'reader-source.py'
    reader_source.write_text('# exact fixture source\n')
    reader = tmp_path/'reader.py'
    reader.write_text('import json\ndef checkpoint_metadata(path):\n    return json.load(open(path))\n')
    policy = {'mellum_checkpoint_metadata_reader_path': str(reader),
        'mellum_checkpoint_metadata_reader_sha256': manager._stable_file_sha256(reader),
        'mellum_checkpoint_metadata_source_path': str(reader_source),
        'mellum_checkpoint_metadata_source_sha256': manager._stable_file_sha256(reader_source)}
    lineage = manager._mellum_checkpoint_lineage(policy, checkpoint, 'd'*64)
    assert lineage['model_identity'] == 'mellum:' + 'c'*64
    assert lineage['global_step'] == 7
    assert lineage['native_snapshot_ancestors'] == ['a'*64, 'b'*64]
    reader.write_text(reader.read_text() + '# changed after policy binding\n')
    with pytest.raises(ValueError, match='source changed'):
        manager._mellum_checkpoint_lineage(policy, checkpoint, 'd'*64)


def test_mellum_private_presence_uses_explicit_sdk_interpreter_and_exact_file_fields(monkeypatch):
    policy = {'mellum_sdk_python': sys.executable}
    receipt = {'run': 'werg/natlang/run', 'name': 'mellum/'+'a'*64+'/checkpoint.pt',
               'identity': {'sha256': 'a'*64, 'bytes': 1234}}
    calls = []
    monkeypatch.setattr(manager.subprocess, 'run', lambda command, **kwargs:
        calls.append((command, kwargs)) or type('Result', (), {
            'returncode': 0, 'stdout': 'NATLANG_PRIVATE_OBJECT_PRESENT=1\n', 'stderr': ''})())
    assert manager._authenticated_private_presence(policy, receipt) is True
    command, kwargs = calls[0]
    assert command[0] == sys.executable and command[1] == '-c'
    assert command[-3:] == [receipt['run'], receipt['name'], '1234']
    assert 'cwctl.load_env_file()' in command[2] and 'wandb.Api(timeout=45)' in command[2]
    assert kwargs['timeout'] == 90 and kwargs['capture_output'] is True
    assert receipt['identity']['sha256'] not in command[2]
