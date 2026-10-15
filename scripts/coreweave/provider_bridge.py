"""External CoreWeave file-protocol bridge; scheduling stays in native_rounds.

The transport is injected so the SDK/account owner can reuse the qualified
private artifact store without moving credentials or large files through DGX.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_hex


ENVELOPE_SCHEMA = 'natlang.native-rl-provider-envelope/1'
REQUEST_SCHEMA = 'natlang.native-rl-bridge-request/1'
RESULT_SCHEMA = 'natlang.native-rl-bridge-result/1'
STOP_SCHEMA = 'natlang.native-rl-bridge-stop/1'


def safe_path(value, root='/work/native-production'):
    path, base = PurePosixPath(value), PurePosixPath(root)
    if not path.is_absolute() or '..' in path.parts or base not in path.parents:
        raise ValueError('provider path escaped its declared namespace')
    return str(path)


def checked_binding(value):
    path = safe_path(value['path'])
    sha, size = value['sha256'], value['bytes']
    if (not isinstance(sha, str) or len(sha) != 64
            or any(c not in '0123456789abcdef' for c in sha)
            or type(size) is not int or size < 0):
        raise ValueError('invalid exact artifact binding')
    return {'path': path, 'sha256': sha, 'bytes': size}


def decode_bound(data, expected):
    expected = checked_binding(expected)
    if len(data) != expected['bytes'] or sha256_hex(data) != expected['sha256']:
        raise ValueError('provider metadata differs from its exact binding')
    return json.loads(data)


def validate_envelope(envelope, request_data, allowed):
    if envelope.get('schema') != ENVELOPE_SCHEMA:
        raise ValueError('unsupported provider envelope')
    request = decode_bound(request_data, envelope['request'])
    identity = request.get('request_id')
    content = {k: v for k, v in request.items() if k != 'request_id'}
    if (request.get('schema') != REQUEST_SCHEMA or request.get('kind') not in {'round', 'refresh'}
            or canonical_json_sha256_hex(content) != identity
            or identity != envelope.get('request_id')):
        raise ValueError('provider request identity/schema changed')
    provider = envelope.get('provider')
    if provider != request.get('provider') or not isinstance(provider, dict):
        raise ValueError('provider binding differs from request')
    target = provider.get('sandbox_id')
    if target not in allowed:
        raise ValueError('request names a sandbox outside the existing resource allowlist')
    safe_path(request['out'])
    safe_path(envelope['result']); safe_path(envelope['stop_file'])
    for binding in input_bindings(request):
        checked_binding(binding)
    return request, target


def input_bindings(request):
    names = ('config', 'state', 'parent_checkpoint', 'semantic_adapter', 'semantic_config')
    result = [request[name] for name in names if request.get(name)]
    result.extend(request.get('updates', {}).values())
    return result


def validate_result(request, request_binding, result):
    if (result.get('schema') != RESULT_SCHEMA or result.get('request_id') != request['request_id']
            or result.get('request_sha256') != request_binding['sha256']
            or result.get('kind') != request['kind']
            or result.get('status') not in {'completed', 'stopped', 'failed'}):
        raise ValueError('provider result is not for this exact request')
    artifacts = result.get('artifacts')
    if not isinstance(artifacts, list):
        raise ValueError('provider result has no artifact inventory')
    seen = {}
    for item in artifacts:
        bound = checked_binding(item)
        safe_path(bound['path'], request['out'])
        if bound['path'] in seen:
            raise ValueError('duplicate provider artifact')
        seen[bound['path']] = bound
    for name in ('state', 'checkpoint', 'deployment'):
        if result.get(name) and checked_binding(result[name]) != seen.get(result[name]['path']):
            raise ValueError('declared provider result is absent from artifact inventory')
    return list(seen.values())


class ProviderBridge:
    """One declared request, with recoverable external operations.

    io supplies read/write metadata, exact artifact mirror, launch/poll/TERM,
    and durable receipt operations. mirror must verify target SHA and size;
    launch must atomically reuse an existing request's process/exit receipt.
    poll returns {terminal, result?}; a missing result after terminal is a
    runtime failure, never a completed learner round.
    """

    def __init__(self, io, coordinator, allowed):
        self.io, self.coordinator, self.allowed = io, coordinator, set(allowed)

    def prepare(self, envelope, *, should_stop=lambda: False):
        raw = self.io.read(self.coordinator, envelope['request']['path'])
        request, target = validate_envelope(envelope, raw, self.allowed)
        key = request['request_id']
        receipt = self.io.receipt(key) or {}
        ownership = {'request': envelope['request'], 'target': target, 'result': envelope['result']}
        if receipt and any(receipt.get(k) != v for k, v in ownership.items()):
            raise ValueError('provider request receipt was rebound')
        if receipt.get('phase') in {'launched', 'stop_sent', 'complete'}:
            return request, target, receipt
        self.io.save_receipt(key, {**ownership, 'phase': 'staging'})
        closure = self.io.dependency_closure(self.coordinator, request)
        for binding in closure["files"]:
            if should_stop():
                return request, target, {**ownership, "phase": "staging_stopped"}
            self.io.mirror(self.coordinator, target, checked_binding(binding))
        self.io.verify_assets(target, closure["assets"])
        if should_stop():
            return request, target, {**ownership, "phase": "staging_stopped"}
        self.io.mirror(self.coordinator, target, checked_binding(envelope['request']))
        # Keep the worker's result private until every result artifact is safe
        # on the coordinator, including when both roles share one sandbox.
        worker_result = '/work/native-production/shared/bridge-workers/' + key + '/result.json'
        process = self.io.launch(target, request, envelope['request']['path'], worker_result)
        receipt = {**ownership, 'phase': 'launched', 'process': process, 'worker_result': worker_result}
        self.io.save_receipt(key, receipt)
        return request, target, receipt

    def tick(self, envelope, *, host_stop=False):
        should_stop = host_stop if callable(host_stop) else lambda: host_stop
        request, target, receipt = self.prepare(envelope, should_stop=should_stop)
        if receipt['phase'] in {'complete', 'staging_stopped'}:
            return True
        stop_data = self.io.read_optional(self.coordinator, envelope['stop_file'])
        if stop_data:
            stop = json.loads(stop_data)
            if (stop.get('schema') != STOP_SCHEMA or stop.get('checkpoint_requested') is not True
                    or stop.get('request') != envelope['request']):
                raise ValueError('provider stop does not belong to the active request')
        stop_requested = host_stop() if callable(host_stop) else host_stop
        if (stop_requested or stop_data) and receipt['phase'] != 'stop_sent':
            # The concrete service verifies PID and /proc start time before TERM.
            self.io.terminate(target, receipt['process'])
            receipt = {**receipt, 'phase': 'stop_sent'}
            self.io.save_receipt(request['request_id'], receipt)
        status = self.io.poll(target, receipt['process'], receipt['worker_result'])
        if not status['terminal']:
            return False
        if not status.get('result'):
            # The owned tree is terminal. Preserve every closed artifact and
            # publish an explicit failed attempt; native_rounds owns retry.
            result_data = json.dumps({
                'schema': RESULT_SCHEMA, 'kind': request['kind'],
                'request_id': request['request_id'],
                'request_sha256': envelope['request']['sha256'],
                'status': 'failed',
                'error': {'type': 'NativeProviderMissingResult',
                          'message': 'owned remote worker exited without its result receipt'},
                'artifacts': self.io.closed_artifacts(target, request['out']),
            }, sort_keys=True).encode()
        else:
            result_data = status['result']
        result = json.loads(result_data)
        for binding in validate_result(request, envelope['request'], result):
            self.io.mirror(target, self.coordinator, binding)
        # The native queue client performs the complete semantic verifier only
        # after these local bytes exist. Its result path is the commit marker.
        self.io.write_atomic(self.coordinator, envelope['result'], result_data)
        self.io.save_receipt(request['request_id'], {**receipt, 'phase': 'complete', 'status': result['status']})
        return True
