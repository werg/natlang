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
    return json.dumps({'model': ferry.TEACHER_SERVE_ALIAS,
        'messages': [{'role': 'user', 'content': 'Write brief natural-language program guidance.'}],
        'max_tokens': 1200, 'temperature': .7,
        'chat_template_kwargs': {'enable_thinking': False}}, separators=(',', ':')).encode()


def teacher_identity():
    return {'sandbox_id': ferry.TEACHER_SANDBOX_ID, 'model_id': ferry.TEACHER_MODEL_ID, 'serve_alias': ferry.TEACHER_SERVE_ALIAS,
        'revision': ferry.TEACHER_REVISION, 'source_bindings': list(ferry.TEACHER_SOURCE_BINDINGS),
        'launch_identity': {'pid': 123, 'start_ticks': 456, 'boot_id': ferry._boot_id(),
            'model': ferry.TEACHER_MODEL_ID, 'revision': ferry.TEACHER_REVISION},
        'download_contract': {'repo': ferry.TEACHER_MODEL_ID,
            'model_path': '/work/generation/qwen36-35b-teacher-20261022/model',
            'revision': ferry.TEACHER_REVISION, 'passed': True, 'members_sha256': 'b' * 64},
        'active_engine': {'model_id': ferry.TEACHER_SERVE_ALIAS, 'source_model_id': ferry.TEACHER_MODEL_ID, 'revision': ferry.TEACHER_REVISION,
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
        self.max_write = 0
        self.files = {}
        self._timed_out = False

    def write_file(self, path, raw):
        self.max_write = max(self.max_write, len(raw))
        if len(raw) > ferry.SDK_CHUNK_BYTES:
            raise AssertionError('fake SDK rejects an over-24-KiB write_file call')
        self.files[path] = bytes(raw)
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(raw)
        return FakeFuture(True)

    def read_file(self, path):
        if path.endswith('.py'):
            return FakeFuture(Path(ferry.__file__).read_bytes())
        if path.endswith('.json') and 'deployments' in path:
            return FakeFuture(ferry.deployment_file_bytes())
        return FakeFuture(Path(path).read_bytes())

    def exec(self, argv, timeout_seconds=60):
        operation = argv[2]
        try:
            if self.role == 'eval' and operation == 'evaluator-next':
                value = self.evaluator_queue.next_manifest()
            elif self.role == 'eval' and operation == 'evaluator-result-assemble':
                ident = argv[3]; total, count, digest = int(argv[5]), int(argv[7]), argv[9]
                raw = ferry.read_chunked(ferry.EVALUATOR_ROOT / 'inbox', ident, total, count, digest)
                ferry._durable_write(ferry._evaluator_result_path(ident), raw)
                value = {'request_id': ident, 'assembled_sha256': digest}
            elif self.role == 'eval' and operation == 'evaluator-deliver':
                ident = argv[3]
                payload = json.loads(ferry._evaluator_result_path(ident).read_bytes())
                response_raw = base64.b64decode(payload['response_b64'], validate=True)
                value = self.evaluator_queue.complete(ident, response_raw, payload['receipt'])
            elif self.role == 'teacher' and operation == 'teacher-assemble':
                ident = argv[3]; total, count, digest = int(argv[5]), int(argv[7]), argv[9]
                raw = ferry.read_chunked(ferry.TEACHER_ROOT / 'inbox', ident, total, count, digest)
                envelope = json.loads(raw)
                result = self.teacher_store.enqueue(envelope)
                if result['state'] == 'queued':
                    self.generated += 1
                    identity = teacher_identity()
                    state, request_bytes, before = self.teacher_store.claim(ident,
                        {'pid': __import__('os').getpid(), 'start_ticks': ferry._proc_start_ticks(__import__('os').getpid()), 'boot_id': ferry._boot_id()})
                    assert state == 'submitted' and request_bytes
                    self.teacher_store.finish(ident, self.response, 200, before, identity)
                    result = self.teacher_store.status(ident)
                if self.timeout_after_enqueue and not self._timed_out:
                    self._timed_out = True
                    raise TimeoutError('SDK timed out after durable teacher result committed')
                value = {'request_id': ident, 'request_sha256': result['request_sha256'], 'state': result['state']}
            elif self.role == 'teacher' and operation == 'teacher-status':
                ident = argv[3]; result = self.teacher_store.status(ident)
                if result is None:
                    value = {'request_id': ident, 'state': 'missing'}
                    return FakeFuture(Result(json.dumps(value, separators=(',', ':'))))
                manifest = None
                if result['state'] in ('complete', 'failed', 'indeterminate'):
                    envelope = {'request_id': ident, 'response_b64': result.get('response_b64') or '',
                                'receipt': result['receipt']}
                    if self.forge:
                        envelope['receipt'] = dict(envelope['receipt'], hmac_sha256='0' * 64)
                    manifest = ferry.write_chunked(ferry.TEACHER_ROOT / 'results', ident,
                                                   ferry.canonical_bytes(envelope))
                value = {'request_id': ident, 'request_sha256': result['request_sha256'],
                         'state': result['state'], 'receipt': result.get('receipt'),
                         'result_manifest': manifest}
            else:
                raise AssertionError((self.role, argv))
            return FakeFuture(Result(json.dumps(value, separators=(',', ':')) if value is not None else 'null'))
        except TimeoutError:
            raise
        except Exception as error:
            return FakeFuture(Result('', str(error), 1))


def setup(monkeypatch, tmp_path, *, forge=False, timeout=False, prompt_size=40, response_size=40):
    monkeypatch.setattr(ferry, 'TEACHER_ROOT', tmp_path / 'teacher')
    monkeypatch.setattr(ferry, 'EVALUATOR_ROOT', tmp_path / 'evaluator')
    monkeypatch.setattr(ferry, 'TEACHER_DB', tmp_path / 'teacher' / 'requests.sqlite3')
    monkeypatch.setattr(ferry, 'EVALUATOR_DB', tmp_path / 'evaluator' / 'requests.sqlite3')
    raw = body()
    if prompt_size != 40:
        request = json.loads(raw); request['messages'][0]['content'] = 'x' * prompt_size
        raw = json.dumps(request, separators=(',', ':')).encode()
    profile = ferry.evaluator_profile('lfm')
    ident = ferry.request_id(ferry.TEACHER_SERVE_ALIAS, raw, profile['sandbox_id'], profile['native_identity'])
    queue = ferry.EvaluatorQueue(ferry.EVALUATOR_DB, KEY, profile['native_identity'], evaluator_sandbox_id=profile['sandbox_id'])
    queue.enqueue(raw, ident)
    store = ferry.TeacherJobStore(ferry.TEACHER_DB, KEY, identity_provider=teacher_identity)
    response = json.dumps({'id': 'chatcmpl-local', 'model': ferry.TEACHER_SERVE_ALIAS,
        'choices': [{'message': {'role': 'assistant', 'content': 'x' * response_size}}]},
        separators=(',', ':')).encode()
    teacher = FakeSandbox('teacher', queue, store, response, forge=forge, timeout_after_enqueue=timeout)
    evaluator = FakeSandbox('eval', queue, store, response)
    monkeypatch.setattr(sdk_ferry, 'PROTOCOL_REMOTE', {'teacher': '/fake/teacher.py', 'evaluator': '/fake/evaluator.py'})
    return ident, queue, store, evaluator, teacher


def test_sdk_restart_reuses_durable_request_without_second_generation(monkeypatch, tmp_path):
    ident, queue, store, evaluator, teacher = setup(monkeypatch, tmp_path, timeout=True, prompt_size=300_000, response_size=400_000)
    with pytest.raises(TimeoutError):
        sdk_ferry.ferry_once(evaluator, teacher)
    assert store.status(ident)['state'] == 'complete'
    result = sdk_ferry.ferry_once(evaluator, teacher)
    assert result == {'request_id': ident, 'state': 'complete'}
    assert teacher.generated == 1
    assert evaluator.max_write <= ferry.SDK_CHUNK_BYTES and teacher.max_write <= ferry.SDK_CHUNK_BYTES
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
    ident = ferry.request_id(ferry.TEACHER_SERVE_ALIAS, raw)
    queue.enqueue(raw, ident)
    changed = raw.replace(b'brief', b'short')
    with pytest.raises(ValueError, match='Idempotency-Key'):
        queue.enqueue(changed, ident)


def test_teacher_transport_timeout_is_terminal_and_never_reposts(tmp_path):
    store = ferry.TeacherJobStore(tmp_path / 'teacher.sqlite3', KEY, identity_provider=teacher_identity)
    raw = body()
    ident = ferry.request_id(ferry.TEACHER_SERVE_ALIAS, raw)
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


def test_sdk_chunks_reassemble_only_complete_hash_bound_payload(tmp_path):
    raw = bytes(range(251)) * 1700
    ident = ferry.request_id(ferry.TEACHER_SERVE_ALIAS, body())
    manifest = ferry.write_chunked(tmp_path, ident, raw)
    assert manifest['total_bytes'] > 155 * 1024
    assert manifest['chunk_count'] > 1
    assert ferry.read_chunked(tmp_path, ident, manifest['total_bytes'], manifest['chunk_count'], manifest['sha256']) == raw
    chunk = ferry._chunk_directory(tmp_path, ident) / '000000.bin'
    chunk.write_bytes(b'forged')
    with pytest.raises(ValueError, match='byte count'):
        ferry.read_chunked(tmp_path, ident, manifest['total_bytes'], manifest['chunk_count'], manifest['sha256'])


def test_deployment_manifest_pins_roles_and_leaves_mellum_unbound():
    assert ferry.evaluator_profile('lfm')['sandbox_id'] == '393f9b75-2666-456c-a850-1b46a8b7d147'
    assert ferry.EVALUATOR_IDS['mellum'] == 'ee341c7b-a425-4b4b-ad7e-f3cf7cbc2369'
    with pytest.raises(ValueError, match='not yet bound'):
        ferry.evaluator_profile('mellum')


def test_download_contract_matches_observed_keys_and_serving_alias_is_separate(tmp_path, monkeypatch):
    download_path = tmp_path / ferry.TEACHER_SOURCE_BINDINGS[2]['path'].lstrip('/')
    download_path.parent.mkdir(parents=True)
    source = {'at': 'redacted', 'members': ['opaque member binding'],
        'model': '/work/generation/qwen36-35b-teacher-20261022/model', 'passed': True,
        'repo': ferry.TEACHER_MODEL_ID, 'revision': ferry.TEACHER_REVISION}
    download_path.write_text(json.dumps(source))
    contract = ferry._download_contract(tmp_path)
    assert contract['repo'] == ferry.TEACHER_MODEL_ID
    assert contract['model_path'] == source['model']
    assert contract['passed'] is True
    assert contract['members_sha256'] == hashlib.sha256(ferry.canonical_bytes(source['members'])).hexdigest()
    source['passed'] = False
    download_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='download receipt differs'):
        ferry._download_contract(tmp_path)

    monkeypatch.setattr(ferry, '_download_contract', lambda: contract)
    models = {'data': [{'id': ferry.TEACHER_SERVE_ALIAS}]}
    launch = {'pid': 123, 'start_ticks': 456, 'boot_id': '12345678-1234-1234-1234-123456789abc',
        'model': ferry.TEACHER_MODEL_ID, 'revision': ferry.TEACHER_REVISION}
    monkeypatch.setattr(ferry, '_launch_contract', lambda: launch)
    monkeypatch.setattr(ferry, '_boot_id', lambda: launch['boot_id'])
    snapshot = ferry.teacher_identity_snapshot(source_bindings=list(ferry.TEACHER_SOURCE_BINDINGS),
        http_json=lambda method, path, **kwargs: (200, json.dumps(models).encode()),
        process_reader=lambda: teacher_identity()['active_engine']['processes'])
    assert snapshot['model_id'] == ferry.TEACHER_MODEL_ID
    assert snapshot['active_engine']['model_id'] == ferry.TEACHER_SERVE_ALIAS
