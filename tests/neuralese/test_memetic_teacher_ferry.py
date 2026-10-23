import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl import memetic_teacher_ferry as ferry
from scripts.coreweave import memetic_teacher_ferry as sdk_ferry


KEY = b'k' * 32


def body():
    return json.dumps({'model': ferry.TEACHER_MODEL_ID,
        'messages': [{'role': 'user', 'content': 'Write brief natural-language program guidance.'}],
        'max_tokens': 1200, 'temperature': .7,
        'chat_template_kwargs': {'enable_thinking': False}}, separators=(',', ':')).encode()


def teacher_identity():
    return {'sandbox_id': ferry.TEACHER_SANDBOX_ID, 'model_id': ferry.TEACHER_MODEL_ID,
        'revision': ferry.TEACHER_REVISION, 'source_bindings': list(ferry.TEACHER_SOURCE_BINDINGS),
        'active_engine': {'model_id': ferry.TEACHER_MODEL_ID,
            'processes': [{'pid': 123, 'start_ticks': 456, 'cmdline_sha256': 'a' * 64}]}}


class Result:
    def __init__(self, stdout='', stderr='', returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


class FakeFuture:
    def __init__(self, value): self.value = value
    def result(self, timeout=None): return self.value


class FakeSandbox:
    def __init__(self, role, evaluator_queue, teacher_store, response, *, forge=False, timeout_after_enqueue=False):
        self.role, self.evaluator_queue, self.teacher_store = role, evaluator_queue, teacher_store
        self.response, self.forge, self.timeout_after_enqueue = response, forge, timeout_after_enqueue
        self.generated = 0
        self.files = {}
        self._timed_out = False

    def write_file(self, path, raw):
        self.files[path] = bytes(raw)
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(raw)
        return FakeFuture(True)

    def read_file(self, path):
        if path.endswith('.py'):
            return FakeFuture(Path(ferry.__file__).read_bytes())
        return FakeFuture(self.files[path])

    def exec(self, argv, timeout_seconds=60):
        operation = argv[2]
        try:
            if self.role == 'eval' and operation == 'evaluator-next':
                value = self.evaluator_queue.next()
            elif self.role == 'eval' and operation == 'evaluator-deliver':
                ident = argv[3]
                remote = ferry.EVALUATOR_ROOT / 'inbox' / (ident + '.json')
                payload = json.loads(remote.read_bytes())
                response_raw = base64.b64decode(payload['response_b64'], validate=True)
                value = self.evaluator_queue.complete(ident, response_raw, payload['receipt'])
            elif self.role == 'teacher' and operation == 'teacher-enqueue':
                ident = argv[3]
                envelope = json.loads((ferry.TEACHER_ROOT / 'inbox' / (ident + '.json')).read_bytes())
                value = self.teacher_store.enqueue(envelope)
                if value['state'] == 'queued':
                    self.generated += 1
                    identity = teacher_identity()
                    state, request_bytes, before = self.teacher_store.claim(ident,
                        {'pid': 1, 'start_ticks': 2, 'boot_id': 'fake'})
                    assert state == 'submitted' and request_bytes
                    receipt = self.teacher_store.finish(ident, self.response, 200, before, identity)
                    value = self.teacher_store.status(ident)
                if self.timeout_after_enqueue and not self._timed_out:
                    self._timed_out = True
                    raise TimeoutError('SDK timed out after remote operation committed')
            elif self.role == 'teacher' and operation == 'teacher-status':
                value = self.teacher_store.status(argv[3])
                if self.forge and value and value.get('receipt'):
                    value['receipt'] = dict(value['receipt'], hmac_sha256='0' * 64)
            else:
                raise AssertionError((self.role, argv))
            if value and isinstance(value, dict) and self.forge and self.role == 'teacher' and operation == 'teacher-enqueue':
                value['receipt'] = dict(value['receipt'], hmac_sha256='0' * 64)
            return FakeFuture(Result(json.dumps(value, separators=(',', ':')) if value is not None else 'null'))
        except TimeoutError:
            raise
        except Exception as error:
            return FakeFuture(Result('', str(error), 1))


def setup(monkeypatch, tmp_path, *, forge=False, timeout=False):
    monkeypatch.setattr(ferry, 'TEACHER_ROOT', tmp_path / 'teacher')
    monkeypatch.setattr(ferry, 'EVALUATOR_ROOT', tmp_path / 'evaluator')
    monkeypatch.setattr(ferry, 'TEACHER_DB', tmp_path / 'teacher' / 'requests.sqlite3')
    monkeypatch.setattr(ferry, 'EVALUATOR_DB', tmp_path / 'evaluator' / 'requests.sqlite3')
    ident = ferry.request_id(ferry.TEACHER_MODEL_ID, body())
    queue = ferry.EvaluatorQueue(ferry.EVALUATOR_DB, KEY)
    queue.enqueue(body(), ident)
    store = ferry.TeacherJobStore(ferry.TEACHER_DB, KEY, identity_provider=teacher_identity)
    raw_response = json.dumps({'id': 'chatcmpl-local', 'model': ferry.TEACHER_MODEL_ID,
        'choices': [{'message': {'role': 'assistant', 'content': '<guidance>Use concise steps.</guidance>'}}]},
        separators=(',', ':')).encode()
    # Teach the host fake worker the same shared key without exposing it to host code.
    teacher = FakeSandbox('teacher', queue, store, raw_response, forge=forge,
                          timeout_after_enqueue=timeout)
    evaluator = FakeSandbox('eval', queue, store, raw_response)
    monkeypatch.setattr(sdk_ferry, 'PROTOCOL_REMOTE', {
        ferry.TEACHER_SANDBOX_ID: '/fake/teacher.py', ferry.EVALUATOR_SANDBOX_ID: '/fake/evaluator.py'})
    return ident, queue, store, evaluator, teacher


def test_sdk_restart_reuses_durable_request_without_second_generation(monkeypatch, tmp_path):
    ident, queue, store, evaluator, teacher = setup(monkeypatch, tmp_path, timeout=True)
    with pytest.raises(TimeoutError):
        sdk_ferry.ferry_once(evaluator, teacher)
    assert store.status(ident)['state'] == 'complete'
    result = sdk_ferry.ferry_once(evaluator, teacher)
    assert result == {'request_id': ident, 'state': 'complete'}
    assert teacher.generated == 1
    assert queue.result(ident)['state'] == 'complete'
    assert hashlib.sha256(queue.result(ident)['response']).hexdigest() == store.status(ident)['response_sha256']


def test_sdk_ferry_refuses_forged_signed_response(monkeypatch, tmp_path):
    ident, queue, store, evaluator, teacher = setup(monkeypatch, tmp_path, forge=True)
    with pytest.raises(RuntimeError, match='fixed ferry operation failed'):
        sdk_ferry.ferry_once(evaluator, teacher)
    assert store.status(ident)['state'] == 'complete'
    assert queue.result(ident)['state'] == 'queued'


def test_request_ids_bind_exact_body_and_reject_rebinding(tmp_path):
    queue = ferry.EvaluatorQueue(tmp_path / 'evaluator.sqlite3', KEY)
    raw = body()
    ident = ferry.request_id(ferry.TEACHER_MODEL_ID, raw)
    queue.enqueue(raw, ident)
    changed = raw.replace(b'brief', b'short')
    with pytest.raises(ValueError, match='Idempotency-Key'):
        queue.enqueue(changed, ident)


def test_teacher_transport_timeout_is_terminal_and_never_reposts(tmp_path):
    store = ferry.TeacherJobStore(tmp_path / 'teacher.sqlite3', KEY, identity_provider=teacher_identity)
    raw = body()
    ident = ferry.request_id(ferry.TEACHER_MODEL_ID, raw)
    envelope = {'schema': ferry.REQUEST_SCHEMA, 'request_id': ident,
        'request_sha256': hashlib.sha256(raw).hexdigest(),
        'request_bytes_b64': base64.b64encode(raw).decode(),
        'evaluator_sandbox_id': ferry.EVALUATOR_SANDBOX_ID,
        'evaluator_identity': ferry.EVALUATOR_NATIVE_IDENTITY}
    store.enqueue(envelope)
    calls = []
    def uncertain(method, path, request, timeout):
        calls.append((method, path, request))
        raise TimeoutError('response lost after submit')
    terminal = ferry.teacher_run(ident, store, http_json=uncertain)
    assert terminal['state'] == 'indeterminate'
    repeated = ferry.teacher_run(ident, store, http_json=uncertain)
    assert repeated['state'] == 'indeterminate'
    assert len(calls) == 1
    assert terminal['receipt']['status'] == 'indeterminate'
    assert terminal['receipt']['response_bytes'] == 0
