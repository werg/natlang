"""SDK-only durable ferry between the pinned native evaluator and Qwen teacher.

Every remote command is a fixed operation of the standalone protocol module.
No endpoint, shell fragment, or remote executable is accepted from CLI input.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
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
    protocol.TEACHER_SANDBOX_ID: '/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry.py',
    protocol.EVALUATOR_SANDBOX_ID: '/work/native-production/control/memetic-teacher-ferry.py',
}


def _result(sandbox, argv, timeout=60):
    result = sandbox.exec(argv, timeout_seconds=timeout).result(timeout=timeout + 30)
    if result.returncode != 0:
        raise RuntimeError(f'fixed ferry operation failed ({result.returncode}): {result.stderr[-1000:]}')
    return result.stdout


def _json_result(sandbox, argv, timeout=60):
    raw = _result(sandbox, argv, timeout)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError('remote ferry operation returned non-JSON output') from error
    return value


def _verify_deployed_protocol(sandbox, remote_path):
    local = Path(protocol.__file__).read_bytes()
    deployed = _read_remote(sandbox, remote_path)
    if hashlib.sha256(deployed).digest() != hashlib.sha256(local).digest():
        raise ValueError('remote ferry protocol code differs from the locally reviewed module bytes')


def _write(sandbox, remote_path, raw):
    result = sandbox.write_file(remote_path, raw).result(timeout=120)
    if result is False:
        raise RuntimeError('SDK refused durable ferry envelope write')


def _read_remote(sandbox, remote_path):
    # SDK file reads preserve response bytes; no shell command is used for reads.
    result = sandbox.read_file(remote_path).result(timeout=120)
    return bytes(result)


def validate_envelope(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {
            'schema', 'request_id', 'request_sha256', 'request_bytes_b64',
            'evaluator_sandbox_id', 'evaluator_identity'}:
        raise ValueError('evaluator queued an unknown teacher request envelope')
    if envelope['schema'] != protocol.REQUEST_SCHEMA or envelope['evaluator_sandbox_id'] != protocol.EVALUATOR_SANDBOX_ID:
        raise ValueError('evaluator request source is outside the exact authorized sandbox')
    if envelope['evaluator_identity'] != protocol.EVALUATOR_NATIVE_IDENTITY:
        raise ValueError('evaluator request is not bound to the authorized native checkpoint')
    body = base64.b64decode(envelope['request_bytes_b64'], validate=True)
    parsed = protocol.validate_chat_body(body)
    if (protocol.request_id(parsed['model'], body) != envelope['request_id']
            or hashlib.sha256(body).hexdigest() != envelope['request_sha256']):
        raise ValueError('evaluator request ID/hash does not bind the exact submitted bytes')
    return body


def ferry_once(evaluator, teacher):
    """Advance one SDK polling pass. Repeats are safe across host restart."""
    _verify_deployed_protocol(evaluator, PROTOCOL_REMOTE[protocol.EVALUATOR_SANDBOX_ID])
    _verify_deployed_protocol(teacher, PROTOCOL_REMOTE[protocol.TEACHER_SANDBOX_ID])
    envelope = _json_result(evaluator, ['python3', PROTOCOL_REMOTE[protocol.EVALUATOR_SANDBOX_ID], 'evaluator-next'])
    if envelope is None:
        return {'state': 'idle'}
    body = validate_envelope(envelope)
    ident = envelope['request_id']
    remote_request = protocol.TEACHER_ROOT / 'inbox' / (ident + '.json')
    _write(teacher, str(remote_request), protocol.canonical_bytes(envelope))
    enqueued = _json_result(teacher, ['python3', PROTOCOL_REMOTE[protocol.TEACHER_SANDBOX_ID], 'teacher-enqueue', ident], timeout=90)
    if enqueued.get('request_id') != ident or enqueued.get('request_sha256') != hashlib.sha256(body).hexdigest():
        raise ValueError('teacher durable queue acknowledged a different request identity')
    status = _json_result(teacher, ['python3', PROTOCOL_REMOTE[protocol.TEACHER_SANDBOX_ID], 'teacher-status', ident], timeout=90)
    if status.get('request_id') != ident or status.get('request_sha256') != envelope['request_sha256']:
        raise ValueError('teacher status is rebound to different request bytes')
    if status.get('state') not in ('complete', 'failed', 'indeterminate'):
        return {'request_id': ident, 'state': status.get('state', 'pending')}
    response_b64 = status.get('response_b64') or ''
    response = base64.b64decode(response_b64, validate=True)
    receipt = status.get('receipt')
    _validate_terminal(receipt, ident, envelope['request_sha256'], response, status['state'])
    result_envelope = protocol.canonical_bytes({'request_id': ident, 'response_b64': response_b64, 'receipt': receipt})
    remote_result = protocol.EVALUATOR_ROOT / 'inbox' / (ident + '.json')
    _write(evaluator, str(remote_result), result_envelope)
    delivered = _json_result(evaluator, ['python3', PROTOCOL_REMOTE[protocol.EVALUATOR_SANDBOX_ID], 'evaluator-deliver', ident], timeout=90)
    if delivered.get('request_id') != ident or delivered.get('state') != status['state']:
        raise ValueError('evaluator durable queue did not accept the same terminal teacher receipt')
    return {'request_id': ident, 'state': delivered['state']}


def _validate_terminal(receipt, ident, request_sha, response, state):
    # The HMAC key remains exclusively on the evaluator and teacher sandboxes.
    # The host validates all public bindings and exact byte hashes; evaluator-deliver
    # then authenticates the signature before changing its durable queue.
    if not isinstance(receipt, dict) or receipt.get('schema') != protocol.RECEIPT_SCHEMA:
        raise ValueError('teacher omitted the signed response receipt')
    if (receipt.get('request_id') != ident or receipt.get('request_sha256') != request_sha
            or receipt.get('status') != state or receipt.get('evaluator_identity') != protocol.EVALUATOR_NATIVE_IDENTITY
            or receipt.get('response_bytes') != len(response)
            or receipt.get('response_sha256') != hashlib.sha256(response).hexdigest()):
        raise ValueError('teacher receipt does not bind the exact request, evaluator, or response bytes')
    teacher = receipt.get('teacher') or {}
    if (teacher.get('sandbox_id') != protocol.TEACHER_SANDBOX_ID
            or teacher.get('model_id') != protocol.TEACHER_MODEL_ID
            or teacher.get('revision') != protocol.TEACHER_REVISION
            or teacher.get('source_bindings') != list(protocol.TEACHER_SOURCE_BINDINGS)):
        raise ValueError('teacher receipt provenance differs from the authorized source/runtime identity')
    if state == 'complete' and not protocol._valid_chat_response(response):
        raise ValueError('teacher result does not contain a valid chat completion')
    if state != 'complete' and response:
        raise ValueError('non-success teacher receipt contains response bytes')


def run(evaluator, teacher, interval=1.0, once=False):
    while True:
        try:
            result = ferry_once(evaluator, teacher)
            if result.get('state') != 'idle':
                print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as error:
            # A failed SDK operation changes no remote terminal state; the next
            # cycle resumes the same durable request ID and never issues a fresh one.
            print(json.dumps({'event': 'ferry_retry', 'error': type(error).__name__}, sort_keys=True), flush=True)
            if once:
                raise
        if once:
            return
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args(argv)
    if not 0.1 <= args.interval <= 60:
        parser.error('--interval must be between 0.1 and 60 seconds')
    evaluator = cwctl.read_sandbox(protocol.EVALUATOR_SANDBOX_ID, patience=60)
    teacher = cwctl.read_sandbox(protocol.TEACHER_SANDBOX_ID, patience=60)
    run(evaluator, teacher, args.interval, args.once)


if __name__ == '__main__':
    main()
