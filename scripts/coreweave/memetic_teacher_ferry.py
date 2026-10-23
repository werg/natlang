"""Bounded-chunk SDK ferry between pinned native evaluators and the Qwen teacher."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'training' / 'neuralese') not in sys.path:
    sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
if str(ROOT / 'scripts' / 'coreweave') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts' / 'coreweave'))
import cwctl
from natlang_neuralese.rl import memetic_teacher_ferry as protocol

PROTOCOL_REMOTE = {
    'teacher': '/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry.py',
    'evaluator': '/work/native-production/control/memetic-teacher-ferry.py',
}
DEPLOYMENTS_REMOTE = {
    'teacher': '/work/generation/qwen36-35b-teacher-20261022/memetic_teacher_ferry_deployments.json',
    'evaluator': '/work/native-production/control/memetic_teacher_ferry_deployments.json',
}


def _result(sandbox, argv, timeout=60):
    result = sandbox.exec(argv, timeout_seconds=timeout).result(timeout=timeout + 30)
    if result.returncode != 0:
        raise RuntimeError(f'fixed ferry operation failed ({result.returncode}): {result.stderr[-1000:]}')
    return result.stdout


def _json_result(sandbox, argv, timeout=60):
    raw = _result(sandbox, argv, timeout)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError('remote ferry operation returned non-JSON output') from error


def _write(sandbox, remote_path, raw):
    if not isinstance(raw, bytes) or len(raw) > protocol.SDK_CHUNK_BYTES:
        raise ValueError('SDK write_file payload exceeds the tested 24 KiB ferry chunk bound')
    result = sandbox.write_file(remote_path, raw).result(timeout=120)
    if result is False:
        raise RuntimeError('SDK refused bounded ferry chunk write')


def _read(sandbox, remote_path):
    value = sandbox.read_file(remote_path).result(timeout=120)
    if not isinstance(value, bytes):
        raise TypeError('SDK ferry chunk read did not return bytes')
    if len(value) > protocol.SDK_CHUNK_BYTES:
        raise ValueError('SDK returned a ferry chunk larger than 24 KiB')
    return value


def _verify_deployment(sandbox, role):
    local_module = Path(protocol.__file__).read_bytes()
    remote_module = sandbox.read_file(PROTOCOL_REMOTE[role]).result(timeout=120)
    if hashlib.sha256(remote_module).digest() != hashlib.sha256(local_module).digest():
        raise ValueError(f'{role} ferry protocol differs from the locally reviewed module bytes')
    local_manifest = protocol.deployment_file_bytes()
    remote_manifest = sandbox.read_file(DEPLOYMENTS_REMOTE[role]).result(timeout=120)
    if hashlib.sha256(remote_manifest).digest() != hashlib.sha256(local_manifest).digest():
        raise ValueError(f'{role} deployment manifest differs from the locally reviewed exact identity bindings')


def _chunk_path(root, ident, index):
    if not protocol.ID_RE.fullmatch(ident) or type(index) is not int or index < 0:
        raise ValueError('invalid ID-addressed SDK chunk path')
    return str(Path(root) / 'sdk-chunks' / ident / f'{index:06d}.bin')


def _read_chunks(sandbox, root, manifest):
    if not isinstance(manifest, dict) or set(manifest) != {'request_id', 'total_bytes', 'chunk_count', 'sha256'}:
        raise ValueError('SDK chunk manifest has an unknown schema')
    ident, total, count, digest = (manifest[k] for k in ('request_id', 'total_bytes', 'chunk_count', 'sha256'))
    if (not protocol.ID_RE.fullmatch(ident) or type(total) is not int or not 0 <= total <= protocol.MAX_TRANSFER_BYTES
            or type(count) is not int or count != max(1, (total + protocol.SDK_CHUNK_BYTES - 1) // protocol.SDK_CHUNK_BYTES)
            or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('SDK chunk manifest exceeds the complete-byte transfer contract')
    pieces = []
    for index in range(count):
        part = _read(sandbox, _chunk_path(root, ident, index))
        expected = protocol.SDK_CHUNK_BYTES if index < count - 1 else total - protocol.SDK_CHUNK_BYTES * (count - 1)
        if len(part) != expected:
            raise ValueError('SDK read returned an incomplete or oversized ferry chunk')
        pieces.append(part)
    raw = b''.join(pieces)
    if len(raw) != total or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('assembled SDK read bytes fail the full payload size/hash')
    return raw


def _write_chunks(sandbox, root, ident, raw):
    if len(raw) > protocol.MAX_TRANSFER_BYTES:
        raise ValueError('SDK ferry payload exceeds the total-transfer bound')
    count = max(1, (len(raw) + protocol.SDK_CHUNK_BYTES - 1) // protocol.SDK_CHUNK_BYTES)
    for index in range(count):
        part = raw[index * protocol.SDK_CHUNK_BYTES:(index + 1) * protocol.SDK_CHUNK_BYTES]
        _write(sandbox, _chunk_path(root, ident, index), part)
    return {'request_id': ident, 'total_bytes': len(raw), 'chunk_count': count,
            'sha256': hashlib.sha256(raw).hexdigest()}


def validate_envelope(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {
            'schema', 'request_id', 'request_sha256', 'request_bytes_b64',
            'evaluator_sandbox_id', 'evaluator_identity'}:
        raise ValueError('evaluator queued an unknown teacher request envelope')
    if envelope['schema'] != protocol.REQUEST_SCHEMA:
        raise ValueError('evaluator request schema is outside the reviewed protocol')
    profile = protocol.evaluator_profile_for(envelope['evaluator_sandbox_id'], envelope['evaluator_identity'])
    body = base64.b64decode(envelope['request_bytes_b64'], validate=True)
    parsed = protocol.validate_chat_body(body)
    if (protocol.request_id(parsed['model'], body, profile['sandbox_id'], profile['native_identity']) != envelope['request_id']
            or hashlib.sha256(body).hexdigest() != envelope['request_sha256']):
        raise ValueError('evaluator request ID/hash does not bind exact bytes and native actor')
    return body, profile


def _validate_terminal(receipt, ident, request_sha, response, state, profile):
    # The HMAC key stays on the two sandboxes. Evaluator-deliver authenticates it.
    if not isinstance(receipt, dict) or receipt.get('schema') != protocol.RECEIPT_SCHEMA:
        raise ValueError('teacher omitted the signed response receipt')
    if (receipt.get('request_id') != ident or receipt.get('request_sha256') != request_sha
            or receipt.get('status') != state or receipt.get('evaluator_sandbox_id') != profile['sandbox_id']
            or receipt.get('evaluator_identity') != profile['native_identity']
            or receipt.get('response_bytes') != len(response)
            or receipt.get('response_sha256') != hashlib.sha256(response).hexdigest()):
        raise ValueError('teacher receipt does not bind exact request, evaluator, or response bytes')
    teacher = receipt.get('teacher') or {}
    if (teacher.get('sandbox_id') != protocol.TEACHER_SANDBOX_ID
            or teacher.get('model_id') != protocol.TEACHER_MODEL_ID
            or teacher.get('serve_alias') != protocol.TEACHER_SERVE_ALIAS
            or teacher.get('revision') != protocol.TEACHER_REVISION
            or teacher.get('source_bindings') != list(protocol.TEACHER_SOURCE_BINDINGS)):
        raise ValueError('teacher receipt provenance differs from its exact source/runtime identity')
    if state != 'failed':
        contract = teacher.get('download_contract') or {}
        if (contract.get('repo') != protocol.TEACHER_MODEL_ID
                or contract.get('model_path') != '/work/generation/qwen36-35b-teacher-20261022/model'
                or contract.get('revision') != protocol.TEACHER_REVISION or contract.get('passed') is not True
                or not re.fullmatch(r'[0-9a-f]{64}', contract.get('members_sha256', ''))):
            raise ValueError('teacher receipt lacks the exact passed repo/revision/model-path/members source contract')
        engine = teacher.get('active_engine') or {}
        launch = teacher.get('launch_identity') or {}
        processes = engine.get('processes') if isinstance(engine, dict) else None
        if (not isinstance(engine, dict) or not isinstance(launch, dict)
                or engine.get('model_id') != protocol.TEACHER_SERVE_ALIAS
                or engine.get('source_model_id') != protocol.TEACHER_MODEL_ID
                or engine.get('revision') != protocol.TEACHER_REVISION
                or launch.get('model') != protocol.TEACHER_MODEL_ID
                or launch.get('revision') != protocol.TEACHER_REVISION
                or type(launch.get('pid')) is not int or launch['pid'] < 1
                or type(launch.get('start_ticks')) is not int or launch['start_ticks'] < 1
                or not re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', launch.get('boot_id', ''))
                or not isinstance(processes, list)
                or not any(isinstance(item, dict) and item.get('pid') == launch.get('pid')
                           and item.get('start_ticks') == launch.get('start_ticks')
                           for item in processes)):
            raise ValueError('teacher receipt active listener differs from the exact source launch PID/start ticks')
    if state == 'complete' and not protocol._valid_chat_response(response):
        raise ValueError('teacher result does not contain a valid chat completion')
    if state != 'complete' and response:
        raise ValueError('non-success teacher receipt contains response bytes')


def ferry_once(evaluator, teacher, student='lfm'):
    """Advance one bounded SDK polling pass; durable queues allow safe restart."""
    profile = protocol.evaluator_profile(student)
    if evaluator_profile_id(evaluator, profile) != profile['sandbox_id']:
        raise ValueError('SDK evaluator object is not bound to the selected allowlisted student sandbox')
    _verify_deployment(evaluator, 'evaluator')
    _verify_deployment(teacher, 'teacher')
    manifest = _json_result(evaluator, ['python3', PROTOCOL_REMOTE['evaluator'], 'evaluator-next', '--student', student])
    if manifest is None:
        return {'state': 'idle'}
    ident = manifest.get('request_id') if isinstance(manifest, dict) else None
    envelope_raw = _read_chunks(evaluator, protocol.EVALUATOR_ROOT / 'requests', manifest)
    envelope = json.loads(envelope_raw)
    body, found_profile = validate_envelope(envelope)
    if found_profile != profile or ident != envelope['request_id']:
        raise ValueError('ferry request was queued under a different evaluator deployment')

    request_sha = hashlib.sha256(body).hexdigest()
    status = _json_result(teacher, ['python3', PROTOCOL_REMOTE['teacher'], 'teacher-status', ident], timeout=90)
    if status.get('request_id') != ident:
        raise ValueError('teacher status is rebound to a different request ID')
    if status.get('state') == 'missing':
        request_manifest = _write_chunks(teacher, protocol.TEACHER_ROOT / 'inbox', ident, envelope_raw)
        status = _json_result(teacher, ['python3', PROTOCOL_REMOTE['teacher'], 'teacher-assemble', ident,
            '--bytes', str(request_manifest['total_bytes']), '--chunks', str(request_manifest['chunk_count']),
            '--sha256', request_manifest['sha256']], timeout=180)
    if status.get('state') not in ('queued', 'submitted', 'running', 'complete', 'failed', 'indeterminate'):
        raise ValueError('teacher queue returned an unknown state')
    status = _json_result(teacher, ['python3', PROTOCOL_REMOTE['teacher'], 'teacher-status', ident], timeout=90)
    if status.get('request_id') != ident or status.get('request_sha256') != request_sha:
        raise ValueError('teacher status is rebound to different request bytes')
    if status.get('state') not in ('complete', 'failed', 'indeterminate'):
        return {'request_id': ident, 'state': status.get('state', 'pending')}
    result_manifest = status.get('result_manifest')
    result_raw = _read_chunks(teacher, protocol.TEACHER_ROOT / 'results', result_manifest)
    result = json.loads(result_raw)
    if not isinstance(result, dict) or set(result) != {'request_id', 'response_b64', 'receipt'} or result['request_id'] != ident:
        raise ValueError('teacher result envelope differs from exact request binding')
    response = base64.b64decode(result['response_b64'], validate=True)
    receipt = result['receipt']
    _validate_terminal(receipt, ident, request_sha, response, status['state'], profile)

    result_manifest = _write_chunks(evaluator, protocol.EVALUATOR_ROOT / 'inbox', ident, result_raw)
    assembled = _json_result(evaluator, [
        'python3', PROTOCOL_REMOTE['evaluator'], 'evaluator-assemble', ident, '--student', student,
        '--bytes', str(result_manifest['total_bytes']), '--chunks', str(result_manifest['chunk_count']),
        '--sha256', result_manifest['sha256']], timeout=180)
    if assembled.get('request_id') != ident or assembled.get('assembled_sha256') != result_manifest['sha256']:
        raise ValueError('evaluator failed to assemble the exact signed teacher receipt')
    delivered = _json_result(evaluator, [
        'python3', PROTOCOL_REMOTE['evaluator'], 'evaluator-deliver', ident, '--student', student], timeout=90)
    if delivered.get('request_id') != ident or delivered.get('state') != status['state']:
        raise ValueError('evaluator durable queue did not accept the same terminal teacher receipt')
    return {'request_id': ident, 'state': delivered['state']}


def evaluator_profile_id(sandbox, profile):
    # Sandbox objects returned by the SDK expose their control-plane ID. Tests
    # may provide a fake object with this exact field; no SDK query is issued.
    ident = getattr(sandbox, 'id', None) or getattr(sandbox, 'sandbox_id', None)
    if ident is None:
        # The SDK object itself was obtained from the fixed read_sandbox argument.
        return profile['sandbox_id']
    return ident


def run(evaluator, teacher, student='lfm', interval=1.0, once=False):
    while True:
        try:
            result = ferry_once(evaluator, teacher, student)
            if result.get('state') != 'idle':
                print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as error:
            print(json.dumps({'event': 'ferry_retry', 'error': type(error).__name__}, sort_keys=True), flush=True)
            if once:
                raise
        if once:
            return
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', choices=tuple(protocol.EVALUATOR_IDS), required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args(argv)
    if not 0.1 <= args.interval <= 60:
        parser.error('--interval must be between 0.1 and 60 seconds')
    profile = protocol.evaluator_profile(args.student)
    evaluator = cwctl.read_sandbox(profile['sandbox_id'], patience=60)
    teacher = cwctl.read_sandbox(protocol.TEACHER_SANDBOX_ID, patience=60)
    run(evaluator, teacher, args.student, args.interval, args.once)


if __name__ == '__main__':
    main()
