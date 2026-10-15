"""Exact artifacts for colocated sandbox rounds and shared semantic refresh.

The provider-specific host launcher owns staging, detached worker PID, polling,
TERM forwarding and artifact fetch. This worker owns no provider resources.
"""
from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..train.native_readiness import _bound

REQUEST_SCHEMA = 'natlang.native-rl-bridge-request/1'
RESULT_SCHEMA = 'natlang.native-rl-bridge-result/1'


def binding(path):
    path = Path(path).absolute()
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path), 'bytes': path.stat().st_size}


def inventory(out, *, exclude=()):
    excluded = {Path(path).absolute() for path in exclude}
    return [binding(path) for path in sorted(Path(out).rglob('*'))
            if path.is_file() and path.absolute() not in excluded]


def write_request(directory, *, kind, out, config=None, state=None, adapter=None,
                  updates=None, semantic_config=None, device='cuda', provider=None):
    directory = Path(directory).absolute(); directory.mkdir(parents=True, exist_ok=True)
    request = {'schema': REQUEST_SCHEMA, 'kind': kind, 'out': str(Path(out).absolute()), 'device': device, 'attempt_id': uuid.uuid4().hex}
    if kind == 'round':
        config_value = json.loads(Path(config).read_text())
        config_copy = directory/('config-' + canonical_json_sha256_hex(config_value) + '.json')
        write_json_durable(config_copy, config_value)
        request['config'] = binding(config_copy)
        request['provider'] = config_value.get('provider')
        current = json.loads(Path(state).read_text()) if state and Path(state).exists() else {}
        adapter = current.get('semantic_adapter', config_value['semantic_adapter'])
        request['parent_checkpoint'] = binding(current.get('checkpoint', config_value['checkpoint']))
        request['native_participant'] = config_value['native_participant']
        if current:
            snapshot = directory/('state-' + canonical_json_sha256_hex(current) + '.json')
            write_json_durable(snapshot, current)
            request['state'] = binding(snapshot)
    elif kind == 'refresh':
        request['updates'] = {name: binding(path) for name, path in updates.items()}
        request['semantic_config'] = binding(semantic_config)
        request['provider'] = provider
    else:
        raise ValueError('native bridge supports only round and refresh requests')
    request['semantic_adapter'] = binding(adapter)
    request['request_id'] = canonical_json_sha256_hex(request)
    destination = directory/(request['request_id'] + '.request.json')
    write_json_durable(destination, request)
    return destination


def read_request(path):
    value = json.loads(Path(path).read_text())
    identity = value.pop('request_id', None)
    if (value.get('schema') != REQUEST_SCHEMA or value.get('kind') not in {'round', 'refresh'}
            or canonical_json_sha256_hex(value) != identity):
        raise ValueError('native bridge request identity/schema changed')
    value['request_id'] = identity
    return value


def verify_result(request_path, result_path):
    request = read_request(request_path)
    result = json.loads(Path(result_path).read_text())
    if (result.get('schema') != RESULT_SCHEMA or result.get('kind') != request['kind']
            or result.get('request_sha256') != sha256_file_hex_cached(request_path)
            or result.get('request_id') != request['request_id']
            or result.get('status') not in {'completed', 'stopped', 'failed'}):
        raise ValueError('native bridge result does not belong to this exact request')
    root = Path(request['out']).absolute()
    artifacts = result.get('artifacts')
    if not isinstance(artifacts, list):
        raise ValueError('native bridge result lacks its durable artifact inventory')
    paths = set()
    for item in artifacts:
        path = Path(item['path']).absolute()
        # Preserve lexical paths of legitimate checkpoint offload symlinks;
        # the bridge fetches their verified bytes to that same output namespace.
        if '..' in path.parts or root not in path.parents or path in paths:
            raise ValueError('native bridge artifact escaped output namespace or was repeated')
        _bound(item)
        if path.stat().st_size != item['bytes']:
            raise ValueError('native bridge artifact size changed')
        paths.add(path)
    for key in ('state', 'checkpoint', 'deployment'):
        if result.get(key) and Path(result[key]['path']).absolute() not in paths:
            raise ValueError('native bridge declared result is absent from fetched artifact inventory: ' + key)
        if result.get(key):
            _bound(result[key])
    if request['kind'] == 'round' and result.get('state'):
        state = json.loads(_bound(result['state']).read_text())
        if sha256_file_hex_cached(state['checkpoint']) != request['parent_checkpoint']['sha256']:
            raise ValueError('native bridge round state substituted its parent policy')
        if state.get('pending_native_checkpoint'):
            if (not result.get('checkpoint') or sha256_file_hex_cached(state['pending_native_checkpoint'])
                    != result['checkpoint']['sha256']):
                raise ValueError('native bridge round lost its actual pending learner checkpoint')
    if request['kind'] == 'refresh' and result.get('deployment'):
        deployment = json.loads(_bound(result['deployment']).read_text())
        if (deployment.get('schema') != 'natlang.native-rl-deployment/1'
                or deployment.get('parent_adapter_sha256') != request['semantic_adapter']['sha256']
                or {name: item['sha256'] for name, item in deployment['updated_participants'].items()}
                    != {name: item['sha256'] for name, item in request['updates'].items()}):
            raise ValueError('native bridge semantic deployment omitted/substituted participant updates')
        if deployment['checkpoint'] != result.get('checkpoint'):
            raise ValueError('native bridge semantic deployment checkpoint differs from its result')
        _bound(deployment['semantic_adapter'])
    return result


def execute(request_path, result_path, *, stop=None):
    from ..train.loop import StopSignal
    owned = stop is None
    if owned:
        stop = StopSignal().install()
    request_path, result_path = Path(request_path), Path(result_path)
    request = read_request(request_path)
    out = Path(request['out']); out.mkdir(parents=True, exist_ok=True)
    result = {'schema': RESULT_SCHEMA, 'kind': request['kind'], 'request_id': request['request_id'],
              'request_sha256': sha256_file_hex_cached(request_path)}
    try:
        adapter_path = _bound(request['semantic_adapter'])
        if request['kind'] == 'round':
            from .production import run
            config_path = _bound(request['config'])
            config = json.loads(config_path.read_text())
            if (not config.get('native_state') or Path(config['out']).absolute() != out.absolute()
                    or config['native_participant'] != request['native_participant']):
                raise ValueError('native bridge round config changes participant/output ownership')
            state_path = out/'state.json'
            if request.get('state'):
                snapshot = _bound(request['state'])
                if state_path.exists() and sha256_file_hex_cached(state_path) != request['state']['sha256']:
                    raise ValueError('remote round state differs: drain prior worker then stage the exact current state')
                if not state_path.exists():
                    write_json_durable(state_path, json.loads(snapshot.read_text()))
            elif state_path.exists():
                raise ValueError('unexpected remote round state; recover its artifacts before starting a new request')
            _bound(request['parent_checkpoint'])
            prior = json.loads(state_path.read_text()) if state_path.exists() else config
            if (sha256_file_hex_cached(prior['checkpoint']) != request['parent_checkpoint']['sha256']
                    or sha256_file_hex_cached(prior.get('semantic_adapter', config['semantic_adapter']))
                        != request['semantic_adapter']['sha256']):
                raise ValueError('native bridge round changed its exact actor checkpoint/adapter')
            state = run(config_path, rounds=1, defer_native_refresh=True, stop=stop)
            result['state'] = binding(state_path)
            if state.get('pending_native_checkpoint'):
                result['checkpoint'] = binding(state['pending_native_checkpoint'])
            result['phase'] = state['phase']
        else:
            from .native_refresh import refresh
            updates = {name: _bound(value) for name, value in request['updates'].items()}
            deployment = refresh(adapter=adapter_path, updates=updates,
                config=_bound(request['semantic_config']), out=out, device=request['device'], stop=stop)
            result['deployment'] = binding(out/'deployment.json')
            result['checkpoint'] = deployment['checkpoint']
        result['status'] = 'stopped' if stop.requested else 'completed'
    except InterruptedError as error:
        result.update(status='stopped', error=str(error))
    except Exception as error:
        result.update(status='stopped' if stop.requested else 'failed',
                      error=type(error).__name__ + ': ' + str(error))
    finally:
        result['artifacts'] = inventory(out, exclude=[result_path])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(result_path, result)
        if owned:
            stop.restore()
    return result


def queued_requests(queue):
    """Small metadata index for the external SDK owner; no credentials here."""
    root = Path(queue)/'requests'
    if not root.exists():
        return []
    result = []
    for path in sorted(root.glob('*.json')):
        value = json.loads(path.read_text())
        if not Path(value['result']).is_file():
            result.append({'envelope': binding(path), **value})
    return result


def enqueue(request_path, result_path, queue, stop_file, *, stop=None):
    """Local filesystem IPC; an external owner performs provider execution."""
    import time
    from ..train.loop import StopSignal
    from ..common.jsonio import utc_now_iso
    request_path, result_path, stop_file = map(lambda path: Path(path).absolute(),
                                              (request_path, result_path, stop_file))
    request = read_request(request_path)
    owned = stop is None
    if owned:
        stop = StopSignal().install()
    root = Path(queue)/'requests'; root.mkdir(parents=True, exist_ok=True)
    envelope_path = root/(request['request_id']+'.json')
    envelope = {'schema': 'natlang.native-rl-provider-envelope/1', 'request': binding(request_path),
        'request_id': request['request_id'], 'provider': request.get('provider'),
        'result': str(result_path), 'stop_file': str(stop_file), 'created_at': utc_now_iso()}
    if envelope_path.exists():
        previous = json.loads(envelope_path.read_text())
        if any(previous[key] != envelope[key] for key in ('request', 'result', 'stop_file')):
            raise ValueError('provider queue request identity was rebound')
    else:
        write_json_durable(envelope_path, envelope)
    try:
        while not result_path.exists():
            if stop.requested and not stop_file.exists():
                write_json_durable(stop_file, {'schema': 'natlang.native-rl-bridge-stop/1',
                    'checkpoint_requested': True, 'request': binding(request_path)})
            # The SDK owner must forward exact-PID TERM and drain checkpoints;
            # exiting this client early would orphan that remote update.
            time.sleep(.2)
        return verify_result(request_path, result_path)
    finally:
        if owned:
            stop.restore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--list-queue', action='store_true')
    args = parser.parse_args(argv)
    if args.list_queue:
        if not args.queue:
            parser.error('--list-queue requires --queue')
        print(json.dumps(queued_requests(args.queue)))
        return 0
    if not args.request or not args.result:
        parser.error('--request and --result are required')
    if args.queue and not args.stop_file:
        parser.error('--queue requires --stop-file')
    result = enqueue(args.request, args.result, args.queue, args.stop_file) if args.queue else execute(args.request, args.result)
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
