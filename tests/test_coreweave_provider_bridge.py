import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cw_provider_bridge', Path(__file__).parents[1] / 'scripts/coreweave/provider_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def payload(value):
    return json.dumps(value, sort_keys=True).encode()


def bound(path, data):
    return {'path': path, 'bytes': len(data), 'sha256': bridge.sha256_hex(data)}


class IO:
    def __init__(self):
        self.files, self.receipts, self.events = {}, {}, []
        self.status = {'terminal': False}

    def read(self, sandbox, path):
        return self.files[sandbox, path]

    def read_optional(self, sandbox, path):
        return self.files.get((sandbox, path))

    def receipt(self, key):
        return self.receipts.get(key)

    def save_receipt(self, key, value):
        self.receipts[key] = value

    def dependency_closure(self, source, request):
        return {'files': bridge.input_bindings(request), 'assets': []}

    def verify_assets(self, target, assets):
        assert not assets

    def mirror(self, source, target, item):
        data = self.read(source, item['path'])
        assert bound(item['path'], data) == item
        self.files[target, item['path']] = data
        self.events.append(('mirror', item['path']))

    def launch(self, target, request, path, result):
        self.events.append(('launch', target))
        return {'pid': 123, 'start_ticks': 456}

    def terminate(self, target, process):
        self.events.append(('term', target, process))

    def poll(self, *args):
        return self.status

    def closed_artifacts(self, target, out):
        return [bound(path, data) for (sid,path),data in self.files.items()
                if sid==target and path.startswith(out+'/')]

    def write_atomic(self, target, path, data):
        self.events.append(('commit', path))
        self.files[target, path] = data


def fixture():
    io = IO()
    prefix = '/work/native-production/shared'
    adapter = bound(prefix + '/adapter.pt', b'adapter')
    request = {'schema': bridge.REQUEST_SCHEMA, 'kind': 'round', 'attempt_id': 'x',
               'out': prefix + '/round', 'device': 'cuda', 'provider': {'sandbox_id': 'lfm'},
               'semantic_adapter': adapter}
    request['request_id'] = bridge.canonical_json_sha256_hex(request)
    raw = payload(request)
    req = bound(prefix + '/request.json', raw)
    io.files['mellum', req['path']] = raw
    io.files['mellum', adapter['path']] = b'adapter'
    envelope = {'schema': bridge.ENVELOPE_SCHEMA, 'request': req, 'request_id': request['request_id'],
                'provider': request['provider'], 'result': prefix + '/result.json', 'stop_file': prefix + '/stop.json'}
    return io, request, envelope


def test_refresh_checkpoint_reference_uses_exact_sized_inventory_without_invented_bytes():
    io, request, envelope = fixture()
    artifact=bound(request['out']+'/checkpoint.pt',b'actual-closed-state')
    result={'schema':bridge.RESULT_SCHEMA,'request_id':request['request_id'],
            'request_sha256':envelope['request']['sha256'],'kind':request['kind'],
            'status':'completed','artifacts':[artifact],
            'checkpoint':{k:v for k,v in artifact.items() if k!='bytes'}}
    assert bridge.validate_result(request,envelope['request'],result)==[artifact]
    for wrong in [{'sha256':'0'*64},{'bytes':artifact['bytes']+1},
                  {'path':request['out']+'/missing.pt'}]:
        changed={**result,'checkpoint':{**result['checkpoint'],**wrong}}
        with pytest.raises(ValueError,match='absent from artifact inventory'):
            bridge.validate_result(request,envelope['request'],changed)


def test_results_are_exposed_only_after_verified_artifacts_and_restart_reuses_worker():
    io, request, envelope = fixture()
    service = bridge.ProviderBridge(io, 'mellum', {'lfm', 'mellum'})
    assert service.tick(envelope) is False
    assert service.tick(envelope) is False
    assert sum(event[0] == 'launch' for event in io.events) == 1
    output = bound(request['out'] + '/checkpoint.pt', b'updated state')
    io.files['lfm', output['path']] = b'updated state'
    result = {'schema': bridge.RESULT_SCHEMA, 'kind': 'round', 'status': 'completed',
              'request_id': request['request_id'], 'request_sha256': envelope['request']['sha256'],
              'artifacts': [output], 'checkpoint': output}
    io.status = {'terminal': True, 'result': payload(result)}
    assert service.tick(envelope) is True
    assert io.events[-2:] == [('mirror', output['path']), ('commit', envelope['result'])]


def test_stop_is_bound_to_request_and_drains_without_result_commit():
    io, request, envelope = fixture()
    service = bridge.ProviderBridge(io, 'mellum', {'lfm'})
    io.files['mellum', envelope['stop_file']] = payload({'schema': bridge.STOP_SCHEMA,
        'checkpoint_requested': True, 'request': envelope['request']})
    assert service.tick(envelope) is False
    assert service.tick(envelope) is False
    assert [e for e in io.events if e[0] == 'term'] == [('term', 'lfm', {'pid': 123, 'start_ticks': 456})]
    assert ('mellum', envelope['result']) not in io.files


def test_foreign_stop_cannot_signal_worker():
    io, _, envelope = fixture()
    io.files['mellum', envelope['stop_file']] = payload({'schema': bridge.STOP_SCHEMA,
        'checkpoint_requested': True, 'request': {**envelope['request'], 'sha256': '0' * 64}})
    with pytest.raises(ValueError, match='stop'):
        bridge.ProviderBridge(io, 'mellum', {'lfm'}).tick(envelope)
    assert not any(e[0] == 'term' for e in io.events)


def test_unowned_resource_and_result_escape_are_rejected():
    io, request, envelope = fixture()
    with pytest.raises(ValueError, match='allowlist'):
        bridge.ProviderBridge(io, 'mellum', {'other'}).tick(envelope)
    result = {'schema': bridge.RESULT_SCHEMA, 'kind': 'round', 'status': 'completed',
              'request_id': request['request_id'], 'request_sha256': envelope['request']['sha256'],
              'artifacts': [bound('/work/native-production/foreign.pt', b'bad')]}
    with pytest.raises(ValueError, match='namespace'):
        bridge.validate_result(request, envelope['request'], result)


def test_missing_terminal_result_commits_failed_attempt_after_closed_checkpoint_ferry():
    io, request, envelope = fixture()
    checkpoint = request['out']+'/checkpoint.pt'
    io.files['lfm', checkpoint] = b'closed recoverable checkpoint'
    io.status = {'terminal': True}
    assert bridge.ProviderBridge(io, 'mellum', {'lfm'}).tick(envelope) is True
    result = json.loads(io.files['mellum',envelope['result']])
    assert result['status'] == 'failed' and result['error']['type'] == 'NativeProviderMissingResult'
    assert io.files['mellum',checkpoint] == b'closed recoverable checkpoint'
    assert io.events[-2:] == [('mirror', checkpoint), ('commit',envelope['result'])]


def test_host_stop_during_staging_does_not_launch_an_unowned_worker():
    io, request, envelope = fixture()
    stopping = [False]
    original = io.mirror
    def mirror(*args):
        original(*args)
        stopping[0] = True
    io.mirror = mirror
    assert bridge.ProviderBridge(io, 'mellum', {'lfm'}).tick(envelope, host_stop=lambda: stopping[0]) is True
    assert not any(event[0] == 'launch' for event in io.events)
    assert io.receipt(request['request_id'])['phase'] == 'staging'
    assert ('mellum', envelope['result']) not in io.files
    # A later supervisor can resume that exact still-queued request.
    assert bridge.ProviderBridge(io, 'mellum', {'lfm'}).tick(envelope) is False
