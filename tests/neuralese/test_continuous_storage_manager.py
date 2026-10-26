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
                  archive_root='/home/data/lfm/rl/cycles', archive_destination='/mnt/external/archive',
                  external_floor_bytes=80*1024**3)
    command = manager._archive_command(policy, 3*1024**3, ['/active/cycle'], apply=True)
    assert '--apply' in command
    assert command[command.index('--archive')+1] == '/mnt/external/archive'
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
