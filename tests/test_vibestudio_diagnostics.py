import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/neuralese_data'))
from vibestudio_diagnostics import retain_instance_diagnostics


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def test_all_instances_and_workspace_streams_remain_source_bound(tmp_path):
    host, capture, output, profile = [tmp_path / name for name in ('host', 'capture', 'diagnostics', 'profile')]
    key = hashlib.sha256(str(host.resolve()).encode()).hexdigest()[:16]
    for case, instance, workspace in [('a', 'first', 'system'), ('b', 'first', 'system'), ('c', 'last', 'personal')]:
        put(capture / 'cases' / case / 'source.json', {'instance': instance, 'workspace': workspace, 'run_id': 'run-' + instance})
    for instance in ('first', 'last', 'unrelated'):
        put(profile / 'instance-state' / key / 'system-test-logs' / (instance + '.log'), 'supervisor:' + instance)
        root = tmp_path / instance
        put(profile / 'dev-instances' / key / (instance + '.json'), {'root': str(root)})
        for workspace in ('system-ws1', 'personal-ws2'):
            put(root / 'workspaces' / workspace / 'state/logs/server-log.jsonl', instance + ':' + workspace)
        put(root / 'workspaces/system-ws1/state/logs/server-log.jsonl.1', 'rotated:' + instance)
    result = retain_instance_diagnostics(capture, host, output, profile)
    assert {group['instance']: len(group['sources']) for group in result['instances']} == {'first': 2, 'last': 1}
    assert not (output / 'instances/unrelated').exists()
    for instance in ('first', 'last'):
        assert (output / 'instances' / instance / 'supervisor.log').read_text() == 'supervisor:' + instance
        for workspace in ('system-ws1', 'personal-ws2'):
            assert (output / 'instances' / instance / 'workspaces' / workspace / 'server-log.jsonl').read_text() == instance + ':' + workspace
    # A later group sample must retain earlier evidence after its live instance retires.
    (profile / 'instance-state' / key / 'system-test-logs/first.log').unlink()
    (profile / 'dev-instances' / key / 'first.json').unlink()
    again = retain_instance_diagnostics(capture, host, output, profile)
    assert (output / 'instances/first/supervisor.log').read_text() == 'supervisor:first'
    assert again['instances'][0]['record_missing'] is True
    assert len(again['instances'][0]['retained']) == 4
    for source in (capture / 'cases').glob('*/source.json'):
        source.unlink()
    retired = retain_instance_diagnostics(capture, host, output, profile)
    assert {group['instance'] for group in retired['instances']} == {'first', 'last'}


def test_missing_exact_source_never_selects_neighbor(tmp_path):
    host, capture, output, profile = [tmp_path / name for name in ('host', 'capture', 'out', 'profile')]
    put(capture / 'cases/a/source.json', {'instance': 'missing'})
    result = retain_instance_diagnostics(capture, host, output, profile)
    assert result['instances'][0]['logs'] == []
    assert result['instances'][0]['record_missing'] is True


@pytest.mark.parametrize('instance', ['../foreign', '/absolute', '', None])
def test_invalid_instance_cannot_escape_owned_log_roots(tmp_path, instance):
    put(tmp_path / 'capture/cases/a/source.json', {'instance': instance})
    with pytest.raises(ValueError, match='Invalid captured instance'):
        retain_instance_diagnostics(tmp_path / 'capture', tmp_path / 'host', tmp_path / 'out', tmp_path / 'profile')


def test_live_log_retirement_during_copy_keeps_previous_snapshot(tmp_path, monkeypatch):
    from vibestudio_diagnostics import _copy
    source, target = tmp_path / 'live.log', tmp_path / 'retained.log'
    source.write_text('new live bytes')
    target.write_text('previous snapshot')
    original = Path.open
    def disappearing(path, *args, **kwargs):
        if path == source:
            source.unlink()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', disappearing)
    assert _copy(source, target) is None
    assert target.read_text() == 'previous snapshot'
    assert not list(tmp_path.glob('retained.log.*'))
