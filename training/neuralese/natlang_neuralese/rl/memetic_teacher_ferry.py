"""Durable, identity-bound relay for the existing self-hosted guidance author.

The evaluator exposes only an OpenAI-compatible chat-completions endpoint on
loopback. Its SQLite queue is ferried by the host SDK; it never opens a route
between sandboxes. The teacher worker submits each request to the one pinned
Qwen server at most once. Ambiguous delivery is recorded as indeterminate.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.server
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


REQUEST_SCHEMA = 'natlang.memetic-teacher-request/1'
RECEIPT_SCHEMA = 'natlang.memetic-teacher-response/1'
ID_RE = re.compile(r'mtf1_[a-f0-9]{64}\Z')
TEACHER_SANDBOX_ID = '78530f71-08fc-4f57-ace9-7fbad64aeeb8'
EVALUATOR_IDS = {
    'lfm': '393f9b75-2666-456c-a850-1b46a8b7d147',
    'mellum': 'ee341c7b-a425-4b4b-ad7e-f3cf7cbc2369',
}
TEACHER_MODEL_ID = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
TEACHER_SERVE_ALIAS = 'qwen3.6-35b-a3b'
TEACHER_REVISION = '1355db6a052410cfd62085d94b58866fd0f2c3c5'
TEACHER_URL = 'http://127.0.0.1:8003'
TEACHER_ROOT = Path('/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry')
EVALUATOR_ROOT = Path('/work/native-production/control/memetic-teacher-ferry')
KEY_PATH = Path('/run/secrets/natlang-memetic-teacher-ferry')
DEPLOYMENTS_PATH = Path(__file__).with_name('memetic_teacher_ferry_deployments.json')
TEACHER_DB = TEACHER_ROOT / 'requests.sqlite3'
EVALUATOR_DB = EVALUATOR_ROOT / 'requests.sqlite3'
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TRANSFER_BYTES = 24 * 1024 * 1024
SDK_CHUNK_BYTES = 24 * 1024
MAX_COMPLETION_TOKENS = 8192

TEACHER_SOURCE_BINDINGS = (
    {'path': '/work/generation/qwen36-35b-teacher-20261022/actor-source-tools-v2.json',
     'sha256': '61a8ca58845063dcd6a25156d0e99bddd5022d8d351b5bd6f062b8f0c1fd2c9a', 'bytes': 989},
    {'path': '/work/generation/qwen36-35b-teacher-20261022/teacher-execution-identity-tools-v2.json',
     'sha256': 'e70a4fd76f2c176e78db366d6113f13df53da1c8f8e3db555b6fedbbf89ab01b', 'bytes': 788},
    {'path': '/work/generation/qwen36-35b-teacher-20261022/download-verified.json',
     'sha256': 'efb3bac317a8bb536e3c1e8d774a0808fc2af580553afc6a42d89b4b1a24e66d', 'bytes': 2609},
    {'path': '/work/generation/qwen36-35b-teacher-20261022/tool-serving-check-v3.json',
     'sha256': 'dd9f6bad17528aabaeeb5b1c32ec7ee38676878f8e5d9f6f6ae3abf00bde1a11', 'bytes': 1329},
    {'path': '/work/generation/qwen36-35b-teacher-20261022/server-launch-tools-v2.json',
     'sha256': '3826dff420cc4be12218bc7ecb4c00b1c72b607d98da68aa92dbb648cabdf75e', 'bytes': 1243},
)
def load_deployments(path: Path = DEPLOYMENTS_PATH):
    value = json.loads(Path(path).read_bytes())
    if (not isinstance(value, dict) or set(value) != {'schema', 'teacher', 'evaluators'}
            or value.get('schema') != 'natlang.memetic-teacher-ferry-deployments/1'
            or not isinstance(value['evaluators'], dict) or set(value['evaluators']) != set(EVALUATOR_IDS)):
        raise ValueError('teacher ferry deployment manifest has an unsupported schema or evaluator role set')
    teacher = value['teacher']
    if teacher != {'sandbox_id': TEACHER_SANDBOX_ID, 'model_id': TEACHER_MODEL_ID,
                   'serve_alias': TEACHER_SERVE_ALIAS, 'revision': TEACHER_REVISION,
                   'source_bindings': list(TEACHER_SOURCE_BINDINGS)}:
        raise ValueError('teacher ferry deployment manifest changed the exact authorized teacher binding')
    for role, profile in value['evaluators'].items():
        if not isinstance(profile, dict) or set(profile) != {'sandbox_id', 'native_identity'}:
            raise ValueError('evaluator deployment profile has unexpected fields')
        if profile['sandbox_id'] != EVALUATOR_IDS[role]:
            raise ValueError('evaluator sandbox ID is outside the exact role allowlist')
        identity = profile['native_identity']
        if identity is not None and (not isinstance(identity, dict)
                or set(identity) != {'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'}
                or any(not isinstance(identity.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', identity[key])
                       for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256'))
                or identity.get('architecture') not in (('lfm2',) if role == 'lfm' else ('mellum',))):
            raise ValueError('evaluator native identity must bind one complete current model actor')
    return value


def evaluator_profile(role: str = 'lfm', deployments=None):
    if role not in EVALUATOR_IDS:
        raise ValueError('evaluator role must be one of the allowlisted student lines')
    value = deployments if deployments is not None else load_deployments()
    profile = value['evaluators'][role]
    if profile['native_identity'] is None:
        raise ValueError(f'{role} evaluator deployment is not yet bound to a current native checkpoint')
    return {'role': role, **profile}


_DEFAULT_PROFILE = evaluator_profile('lfm')
EVALUATOR_SANDBOX_ID = _DEFAULT_PROFILE['sandbox_id']
EVALUATOR_NATIVE_IDENTITY = _DEFAULT_PROFILE['native_identity']


def deployment_file_bytes() -> bytes:
    return DEPLOYMENTS_PATH.read_bytes()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def request_id(model: str, body: bytes, evaluator_sandbox_id: str = EVALUATOR_SANDBOX_ID,
               evaluator_identity: dict = EVALUATOR_NATIVE_IDENTITY) -> str:
    if model != TEACHER_SERVE_ALIAS or not isinstance(body, bytes):
        raise ValueError('teacher request ID requires the exact authorised model and raw request bytes')
    profile = evaluator_profile_for(evaluator_sandbox_id, evaluator_identity)
    binding = canonical_bytes({'evaluator_sandbox_id': evaluator_sandbox_id,
                               'evaluator_identity': profile['native_identity']})
    return 'mtf1_' + hashlib.sha256(b'natlang.memetic-teacher-call/2\0' + binding + b'\0' + model.encode() + b'\0' + body).hexdigest()


def evaluator_profile_for(sandbox_id: str, identity: dict, deployments=None):
    manifest = deployments if deployments is not None else load_deployments()
    for role in ('lfm', 'mellum'):
        profile = manifest['evaluators'][role]
        if profile['sandbox_id'] == sandbox_id and profile['native_identity'] == identity and identity is not None:
            return {'role': role, **profile}
    raise ValueError('evaluator sandbox/checkpoint pair is not provisioned in the exact deployment manifest')


def validate_chat_body(body: bytes) -> dict:
    if not isinstance(body, bytes) or len(body) > MAX_REQUEST_BYTES:
        raise ValueError('author request exceeds the bounded chat request size')
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('author request is not valid JSON') from error
    if not isinstance(value, dict) or set(value) - {'model', 'messages', 'max_tokens', 'temperature',
                                                   'chat_template_kwargs'}:
        raise ValueError('author request contains unsupported chat-completions fields')
    if value.get('model') != TEACHER_SERVE_ALIAS:
        raise ValueError('author request model is outside the exact teacher allowlist')
    messages = value.get('messages')
    if (not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict)
            or set(messages[0]) != {'role', 'content'} or messages[0].get('role') != 'user'
            or not isinstance(messages[0].get('content'), str) or not messages[0]['content'].strip()):
        raise ValueError('author requests contain one nonempty natural-language user prompt')
    if type(value.get('max_tokens')) is not int or not 1 <= value['max_tokens'] <= MAX_COMPLETION_TOKENS:
        raise ValueError('author max_tokens must be an integer inside the teacher request bound')
    temperature = value.get('temperature')
    if type(temperature) not in (int, float) or not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 2:
        raise ValueError('author temperature must be finite and in [0, 2]')
    if value.get('chat_template_kwargs') != {'enable_thinking': False}:
        raise ValueError('author request must use the pinned non-thinking chat rendering')
    return value


def _connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    connection.execute('PRAGMA busy_timeout=30000')
    connection.execute('''CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY,
        request_sha256 TEXT NOT NULL,
        request_bytes BLOB NOT NULL,
        evaluator_identity TEXT NOT NULL,
        evaluator_sandbox_id TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL,
        response_sha256 TEXT,
        response_bytes BLOB,
        response_status INTEGER,
        receipt_json TEXT,
        error_json TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )''')
    columns = {row[1] for row in connection.execute('PRAGMA table_info(requests)')}
    if 'evaluator_sandbox_id' not in columns:
        connection.execute("ALTER TABLE requests ADD COLUMN evaluator_sandbox_id TEXT NOT NULL DEFAULT ''")
    return connection


class EvaluatorQueue:
    """Durable evaluator-side request queue; result acceptance requires a teacher HMAC."""

    def __init__(self, path: Path, key: bytes, evaluator_identity: dict | None = None,
                 identity_provider: Callable[[], dict] | None = None, evaluator_sandbox_id=EVALUATOR_SANDBOX_ID):
        if len(key) < 32:
            raise ValueError('teacher ferry HMAC key must be at least 256 bits')
        self.path, self.key = Path(path), key
        self.evaluator_sandbox_id = evaluator_sandbox_id
        profile = evaluator_profile_for(evaluator_sandbox_id, evaluator_identity or EVALUATOR_NATIVE_IDENTITY)
        self.evaluator_identity = evaluator_identity or profile['native_identity']
        self.identity_provider = identity_provider
        self._initialize()

    def _initialize(self):
        with _connect(self.path) as db:
            db.execute('''CREATE INDEX IF NOT EXISTS requests_state_order
                          ON requests(state, created_at, request_id)''')

    def enqueue(self, raw: bytes, claimed_id: str) -> dict:
        value = validate_chat_body(raw)
        expected = request_id(value['model'], raw, self.evaluator_sandbox_id, self.evaluator_identity)
        if claimed_id != expected or not ID_RE.fullmatch(claimed_id):
            raise ValueError('Idempotency-Key does not bind these exact request bytes')
        actual_identity = self.identity_provider() if self.identity_provider else self.evaluator_identity
        if actual_identity != self.evaluator_identity:
            raise ValueError('active evaluator identity changed from the pinned native checkpoint')
        digest = hashlib.sha256(raw).hexdigest()
        identity_raw = canonical_bytes(actual_identity).decode()
        now = time.time()
        with _connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT request_sha256,request_bytes,evaluator_identity,state,response_sha256,evaluator_sandbox_id '
                             'FROM requests WHERE request_id=?', (claimed_id,)).fetchone()
            if row is None:
                db.execute('INSERT INTO requests(request_id,request_sha256,request_bytes,evaluator_identity,evaluator_sandbox_id,state,created_at,updated_at) '
                           'VALUES(?,?,?,?,?,?,?,?)', (claimed_id, digest, raw, identity_raw, self.evaluator_sandbox_id, 'queued', now, now))
                state = 'queued'
            else:
                if row[0] != digest or bytes(row[1]) != raw or row[2] != identity_raw or row[5] != self.evaluator_sandbox_id:
                    raise ValueError('durable request ID was rebound to different bytes or evaluator identity')
                state = row[3]
            db.execute('COMMIT')
        return {'request_id': claimed_id, 'request_sha256': digest, 'state': state}

    def next(self) -> dict | None:
        with _connect(self.path) as db:
            row = db.execute("SELECT request_id,request_sha256,request_bytes,evaluator_identity,state,evaluator_sandbox_id FROM requests "
                             "WHERE state='queued' ORDER BY created_at,request_id LIMIT 1").fetchone()
        if row is None:
            return None
        return {'schema': REQUEST_SCHEMA, 'request_id': row[0], 'request_sha256': row[1],
                'request_bytes_b64': base64.b64encode(bytes(row[2])).decode(),
                'evaluator_sandbox_id': row[5],
                'evaluator_identity': json.loads(row[3])}

    def next_manifest(self) -> dict | None:
        envelope = self.next()
        if envelope is None:
            return None
        raw = canonical_bytes(envelope)
        manifest = write_chunked(EVALUATOR_ROOT / 'requests', envelope['request_id'], raw)
        return manifest

    def complete(self, request_id_value: str, response: bytes, receipt: dict) -> dict:
        if not ID_RE.fullmatch(request_id_value):
            raise ValueError('invalid durable teacher request ID')
        validate_teacher_receipt(request_id_value, response, receipt, self.key)
        response_sha = hashlib.sha256(response).hexdigest()
        current_identity = self.identity_provider() if self.identity_provider else self.evaluator_identity
        if current_identity != self.evaluator_identity:
            raise ValueError('active evaluator identity changed before teacher result delivery')
        with _connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT request_sha256,state,response_sha256,response_bytes,evaluator_identity,evaluator_sandbox_id FROM requests WHERE request_id=?',
                             (request_id_value,)).fetchone()
            if row is None:
                raise ValueError('teacher result has no exact durable evaluator request')
            if receipt['request_sha256'] != row[0]:
                raise ValueError('teacher result belongs to different request bytes')
            if (row[4] != canonical_bytes(current_identity).decode()
                    or row[5] != self.evaluator_sandbox_id
                    or receipt.get('evaluator_sandbox_id') != self.evaluator_sandbox_id):
                raise ValueError('teacher result belongs to a different evaluator checkpoint binding')
            if row[1] == 'complete':
                if row[2] != response_sha or bytes(row[3]) != response:
                    raise ValueError('completed request refuses a conflicting or forged second response')
                db.execute('COMMIT')
                return {'request_id': request_id_value, 'state': 'complete', 'reused': True}
            if row[1] == 'indeterminate':
                raise ValueError('indeterminate request cannot be completed by a later response')
            if receipt['status'] in ('failed', 'indeterminate'):
                state = receipt['status']
                db.execute('UPDATE requests SET state=?,error_json=?,receipt_json=?,updated_at=? WHERE request_id=?',
                           (state, canonical_bytes(receipt.get('error') or {}).decode(),
                            canonical_bytes(receipt).decode(), time.time(), request_id_value))
                db.execute('COMMIT')
                return {'request_id': request_id_value, 'state': state, 'reused': False}
            response_status = receipt.get('response_status')
            if response_status != 200 or not _valid_chat_response(response):
                raise ValueError('signed teacher result is not a successful chat-completions response')
            db.execute('UPDATE requests SET state=?,response_sha256=?,response_bytes=?,response_status=?,receipt_json=?,updated_at=? '
                       'WHERE request_id=?', ('complete', response_sha, response, response_status,
                        canonical_bytes(receipt).decode(), time.time(), request_id_value))
            db.execute('COMMIT')
        return {'request_id': request_id_value, 'state': 'complete', 'reused': False}

    def indeterminate(self, request_id_value: str, receipt: dict) -> dict:
        return self.complete(request_id_value, b'', receipt)

    def result(self, request_id_value: str):
        with _connect(self.path) as db:
            row = db.execute('SELECT state,response_bytes,response_status,error_json FROM requests WHERE request_id=?',
                             (request_id_value,)).fetchone()
        if row is None:
            return None
        state, payload, status, error = row
        return {'state': state, 'response': bytes(payload) if payload is not None else None,
                'response_status': status, 'error': json.loads(error) if error else None}


def _valid_chat_response(raw: bytes) -> bool:
    if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
        return False
    try:
        response = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    choices = response.get('choices') if isinstance(response, dict) else None
    return (isinstance(choices, list) and bool(choices) and isinstance(choices[0], dict)
            and isinstance(choices[0].get('message'), dict)
            and isinstance(choices[0]['message'].get('content'), str))


def _receipt_unsigned(receipt: dict) -> dict:
    return {key: value for key, value in receipt.items() if key != 'hmac_sha256'}


def sign_receipt(receipt: dict, key: bytes) -> dict:
    unsigned = _receipt_unsigned(receipt)
    signature = hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    return {**unsigned, 'hmac_sha256': signature}


def validate_teacher_receipt(request_id_value: str, response: bytes, receipt: dict, key: bytes):
    allowed = {'schema', 'request_id', 'request_sha256', 'status', 'response_status',
               'response_sha256', 'response_bytes', 'teacher', 'evaluator_sandbox_id',
               'evaluator_identity', 'created_at', 'error', 'hmac_sha256'}
    if not isinstance(receipt, dict) or set(receipt) - allowed:
        raise ValueError('teacher response receipt has unsupported fields')
    signature = receipt.get('hmac_sha256')
    expected = hmac.new(key, canonical_bytes(_receipt_unsigned(receipt)), hashlib.sha256).hexdigest()
    if (not isinstance(signature, str) or not hmac.compare_digest(signature, expected)
            or receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('request_id') != request_id_value
            or receipt.get('status') not in ('complete', 'failed', 'indeterminate')):
        raise ValueError('teacher response receipt signature/schema/request identity failed')
    if (not isinstance(receipt.get('request_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['request_sha256'])):
        raise ValueError('teacher response receipt lacks exact request bytes hash')
    if len(response) != receipt.get('response_bytes') or hashlib.sha256(response).hexdigest() != receipt.get('response_sha256'):
        raise ValueError('teacher response bytes differ from signed exact byte count/hash')
    try:
        evaluator_profile_for(receipt.get('evaluator_sandbox_id'), receipt.get('evaluator_identity'))
    except ValueError as error:
        raise ValueError('teacher response is not bound to an allowlisted native evaluator checkpoint') from error
    teacher = receipt.get('teacher')
    if (not isinstance(teacher, dict) or teacher.get('sandbox_id') != TEACHER_SANDBOX_ID
            or teacher.get('model_id') != TEACHER_MODEL_ID or teacher.get('serve_alias') != TEACHER_SERVE_ALIAS
            or teacher.get('revision') != TEACHER_REVISION
            or teacher.get('source_bindings') != list(TEACHER_SOURCE_BINDINGS)):
        raise ValueError('teacher response provenance is outside the exact authorized source binding')
    if receipt['status'] != 'failed':
        contract = teacher.get('download_contract')
        if (not isinstance(contract, dict)
                or contract.get('repo') != TEACHER_MODEL_ID
                or contract.get('model_path') != '/work/generation/qwen36-35b-teacher-20261022/model'
                or contract.get('revision') != TEACHER_REVISION or contract.get('passed') is not True
                or not re.fullmatch(r'[0-9a-f]{64}', contract.get('members_sha256', ''))):
            raise ValueError('teacher receipt lacks its observed repository/revision/path/members contract')
        engine = teacher.get('active_engine')
        launch = teacher.get('launch_identity')
        if (not isinstance(engine, dict) or engine.get('model_id') != TEACHER_SERVE_ALIAS
                or engine.get('source_model_id') != TEACHER_MODEL_ID or engine.get('revision') != TEACHER_REVISION
                or not isinstance(launch, dict) or launch.get('model') != TEACHER_MODEL_ID
                or launch.get('revision') != TEACHER_REVISION
                or type(launch.get('pid')) is not int or launch['pid'] < 1
                or type(launch.get('start_ticks')) is not int or launch['start_ticks'] < 1
                or not isinstance(launch.get('boot_id'), str)
                or not isinstance(engine.get('processes'), list) or not engine['processes']
                or any(type(item.get('pid')) is not int or item['pid'] < 1
                       or type(item.get('start_ticks')) is not int or item['start_ticks'] < 1
                       or not re.fullmatch(r'[0-9a-f]{64}', item.get('cmdline_sha256', ''))
                       for item in engine['processes'] if isinstance(item, dict))
                or any(not isinstance(item, dict) for item in engine['processes'])):
            raise ValueError('teacher response is not bound to an observed active engine PID/checkpoint')
        _validate_launch_process(launch, engine['processes'], launch.get('boot_id') if isinstance(launch, dict) else '')
    if receipt['status'] == 'complete' and (receipt.get('response_status') != 200 or not _valid_chat_response(response)):
        raise ValueError('teacher response receipt claims success for a non-chat response')
    if receipt['status'] in ('failed', 'indeterminate') and response:
        raise ValueError('non-success teacher receipt cannot contain fabricated completion bytes')


def load_key(path: Path = KEY_PATH) -> bytes:
    path = Path(path)
    mode = path.stat().st_mode
    if mode & 0o077:
        raise ValueError('teacher ferry key is readable or writable by group/other')
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError('teacher ferry key must contain at least 32 random bytes')
    return key


class _EvaluatorHandler(http.server.BaseHTTPRequestHandler):
    server_version = 'natlang-memetic-teacher-ferry/1'

    @property
    def queue(self) -> EvaluatorQueue:
        return self.server.queue  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        # Prompts, responses, IDs and signatures stay out of process logs.
        return

    def _reply(self, status: int, raw: bytes, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/health':
            self._reply(200, b'{"ok":true,"service":"memetic-teacher-ferry"}')
            return
        self._reply(404, b'{"error":{"code":"not-found"}}')

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self._reply(404, b'{"error":{"code":"not-found"}}')
            return
        try:
            length = int(self.headers.get('Content-Length', '-1'))
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError('author request content length is outside the bounded range')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('author request body was truncated')
            claimed = self.headers.get('Idempotency-Key')
            if not claimed:
                parsed = validate_chat_body(raw)
                claimed = request_id(parsed['model'], raw, self.queue.evaluator_sandbox_id, self.queue.evaluator_identity)
            row = self.queue.enqueue(raw, claimed)
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                result = self.queue.result(row['request_id'])
                if result and result['state'] == 'complete':
                    self._reply(result['response_status'], result['response'], 'application/json')
                    return
                if result and result['state'] == 'indeterminate':
                    raw_error = canonical_bytes({'error': {'code': 'memetic-author-indeterminate',
                        'request_id': row['request_id'], 'detail': 'teacher submission outcome is indeterminate; this request was not replayed'}})
                    self._reply(502, raw_error)
                    return
                if result and result['state'] == 'failed':
                    raw_error = canonical_bytes({'error': {'code': 'memetic-author-failed',
                        'request_id': row['request_id'], 'detail': 'teacher request was rejected before submission; see the bound ferry receipt'}})
                    self._reply(502, raw_error)
                    return
                time.sleep(.25)
            self._reply(504, canonical_bytes({'error': {'code': 'memetic-author-pending',
                'request_id': row['request_id'], 'detail': 'request remains queued under this idempotency key'}}))
        except (ValueError, KeyError, TypeError) as error:
            self._reply(400, canonical_bytes({'error': {'code': 'memetic-author-request', 'detail': str(error)}}))


def evaluator_native_identity(student='lfm'):
    request = urllib.request.Request('http://127.0.0.1:8765/v1/neuralese/info', method='GET')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=20) as response:
        raw = response.read(1024 * 1024)
    info = json.loads(raw)
    if not (info.get('native_state') or {}).get('enabled'):
        raise ValueError('native evaluator state is not enabled')
    identity = info.get('native_identity')
    profile = evaluator_profile(student)
    if identity != profile['native_identity']:
        raise ValueError('active evaluator checkpoint/tokenizer/embedding identity differs from the pinned native actor')
    return identity


def evaluator_server(path: Path = EVALUATOR_DB, key_path: Path = KEY_PATH,
                     identity: dict | None = None, host='127.0.0.1', port=8766, student='lfm'):
    if host != '127.0.0.1':
        raise ValueError('memetic teacher ferry is loopback-only')
    profile = evaluator_profile(student)
    actual = evaluator_native_identity(student)
    if identity is not None and identity != actual:
        raise ValueError('configured evaluator identity differs from active native evaluator')
    queue = EvaluatorQueue(path, load_key(key_path), actual,
        identity_provider=lambda: evaluator_native_identity(student), evaluator_sandbox_id=profile['sandbox_id'])
    server = http.server.ThreadingHTTPServer((host, port), _EvaluatorHandler)
    server.daemon_threads = True
    server.queue = queue
    return server


def _proc_start_ticks(pid: int, proc_root=Path('/proc')) -> int:
    fields = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[19])


def _listening_socket_inodes(port: int, proc_root=Path('/proc')) -> set[str]:
    inodes = set()
    for name in ('net/tcp', 'net/tcp6'):
        path = proc_root / name
        if not path.is_file():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 9 and fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def active_engine_processes(port=8003, proc_root=Path('/proc')) -> list[dict]:
    """Bind the teacher response to live PIDs that own the exact local listen socket."""
    inodes = _listening_socket_inodes(port, proc_root)
    if not inodes:
        raise RuntimeError('exact teacher HTTP port has no live listening socket')
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            command = (entry / 'cmdline').read_bytes()
            if b'vllm' not in command.lower() or str(port).encode() not in command:
                continue
            owned = False
            for fd in (entry / 'fd').iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if link.startswith('socket:[') and link[8:-1] in inodes:
                    owned = True
                    break
            if not owned:
                continue
            found.append({'pid': pid, 'start_ticks': _proc_start_ticks(pid, proc_root),
                          'cmdline_sha256': hashlib.sha256(command).hexdigest()})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    found.sort(key=lambda row: row['pid'])
    if not found:
        raise RuntimeError('teacher listen socket is not owned by an identifiable active vLLM engine PID')
    return found


def _http_json(method: str, path: str, body: bytes | None = None, timeout=30):
    request = urllib.request.Request(TEACHER_URL + path, data=body, method=method,
                                     headers={'content-type': 'application/json'} if body is not None else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError('teacher response exceeds the bounded response size')
            return response.status, raw
    except urllib.error.HTTPError as error:
        raw = error.read(MAX_RESPONSE_BYTES + 1)
        return error.code, raw


def _download_contract(root=Path('/')) -> dict:
    path = root / TEACHER_SOURCE_BINDINGS[2]['path'].lstrip('/')
    download = json.loads(path.read_bytes())
    if (not isinstance(download, dict) or set(download) != {'at', 'members', 'model', 'passed', 'repo', 'revision'}
            or download.get('model') != '/work/generation/qwen36-35b-teacher-20261022/model'
            or download.get('repo') != TEACHER_MODEL_ID or download.get('revision') != TEACHER_REVISION
            or download.get('passed') is not True
            or not isinstance(download.get('members'), (list, dict)) or not download['members']):
        raise ValueError('teacher download receipt differs from its exact observed repository/path/revision/members contract')
    return {'repo': download['repo'], 'model_path': download['model'],
            'revision': download['revision'], 'passed': download['passed'],
            'members_sha256': hashlib.sha256(canonical_bytes(download['members'])).hexdigest()}


def _launch_contract(root=Path('/')) -> dict:
    path = root / TEACHER_SOURCE_BINDINGS[4]['path'].lstrip('/')
    launch = json.loads(path.read_bytes())
    expected_keys = {'at', 'argv', 'model', 'revision', 'policy', 'pid', 'start_ticks',
                     'boot_id', 'tool_call_parser', 'previous_owners_exited'}
    if (not isinstance(launch, dict) or set(launch) != expected_keys
            or launch.get('model') != TEACHER_MODEL_ID or launch.get('revision') != TEACHER_REVISION
            or not isinstance(launch.get('argv'), list) or not launch['argv']
            or type(launch.get('pid')) is not int or launch['pid'] < 1
            or type(launch.get('start_ticks')) is not int or launch['start_ticks'] < 1
            or not isinstance(launch.get('boot_id'), str)
            or not re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', launch['boot_id'])
            or type(launch.get('previous_owners_exited')) is not bool):
        raise ValueError('teacher launch receipt differs from the exact model/revision/process identity contract')
    return {key: launch[key] for key in ('pid', 'start_ticks', 'boot_id', 'model', 'revision')}


def _validate_launch_process(launch: dict, processes: list[dict], current_boot_id: str):
    if (not isinstance(launch, dict) or launch.get('boot_id') != current_boot_id
            or not any(item.get('pid') == launch.get('pid') and item.get('start_ticks') == launch.get('start_ticks')
                       for item in processes if isinstance(item, dict))):
        raise ValueError('active Qwen listener PID/start ticks do not match the pinned launch receipt')


def verify_teacher_source_files(root=Path('/')) -> list[dict]:
    verified = []
    for binding in TEACHER_SOURCE_BINDINGS:
        path = root / binding['path'].lstrip('/')
        raw = path.read_bytes()
        if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
            raise ValueError('teacher actor/runtime/source receipt differs from exact pinned provenance: ' + binding['path'])
        verified.append(dict(binding))
    _download_contract(root)
    return verified


def teacher_identity_snapshot(source_bindings=None, *, http_json=_http_json,
                              process_reader=active_engine_processes) -> dict:
    sources = source_bindings if source_bindings is not None else verify_teacher_source_files()
    download_contract = _download_contract()
    status, raw = http_json('GET', '/v1/models', timeout=30)
    if status != 200:
        raise RuntimeError('pinned teacher models endpoint is not healthy')
    value = json.loads(raw)
    rows = value.get('data') if isinstance(value, dict) else None
    if not isinstance(rows, list) or not any(isinstance(item, dict) and item.get('id') == TEACHER_SERVE_ALIAS for item in rows):
        raise ValueError('active teacher API does not expose the pinned Qwen serving alias')
    processes = process_reader()
    launch_identity = _launch_contract()
    _validate_launch_process(launch_identity, processes, _boot_id())
    return {'sandbox_id': TEACHER_SANDBOX_ID, 'model_id': TEACHER_MODEL_ID,
            'serve_alias': TEACHER_SERVE_ALIAS, 'revision': TEACHER_REVISION,
            'source_bindings': list(sources), 'download_contract': download_contract,
            'launch_identity': launch_identity,
            'active_engine': {'model_id': TEACHER_SERVE_ALIAS, 'source_model_id': TEACHER_MODEL_ID,
                              'revision': TEACHER_REVISION, 'processes': processes}}


def _boot_id(proc_root=Path('/proc')) -> str:
    return (proc_root / 'sys/kernel/random/boot_id').read_text().strip()


def _owned_process_alive(owner: dict, proc_root=Path('/proc')) -> bool:
    try:
        if owner.get('boot_id') != _boot_id(proc_root):
            return False
        pid = owner.get('pid')
        if type(pid) is not int or _proc_start_ticks(pid, proc_root) != owner.get('start_ticks'):
            return False
        stat = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z'
    except (OSError, ValueError, TypeError):
        return False


class TeacherJobStore:
    """Teacher-side persistent at-most-once submissions and signed results."""

    def __init__(self, path: Path = TEACHER_DB, key: bytes | None = None,
                 identity_provider: Callable[[], dict] = teacher_identity_snapshot):
        self.path, self.key = Path(path), key if key is not None else load_key()
        if len(self.key) < 32:
            raise ValueError('teacher ferry HMAC key must be at least 256 bits')
        self.identity_provider = identity_provider

    def enqueue(self, envelope: dict) -> dict:
        if (not isinstance(envelope, dict) or set(envelope) != {'schema', 'request_id','request_sha256',
                'request_bytes_b64','evaluator_sandbox_id','evaluator_identity'}
                or envelope.get('schema') != REQUEST_SCHEMA
                or envelope.get('evaluator_sandbox_id') not in EVALUATOR_IDS.values()):
            raise ValueError('teacher enqueue is outside the exact evaluator and checkpoint binding')
        profile = evaluator_profile_for(envelope['evaluator_sandbox_id'], envelope['evaluator_identity'])
        ident = envelope['request_id']
        try:
            raw = base64.b64decode(envelope['request_bytes_b64'], validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError('teacher request bytes are not exact base64') from error
        value = validate_chat_body(raw)
        if request_id(value['model'], raw, envelope['evaluator_sandbox_id'], profile['native_identity']) != ident or hashlib.sha256(raw).hexdigest() != envelope['request_sha256']:
            raise ValueError('teacher request ID/hash does not match the exact raw chat request')
        now = time.time()
        with _connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT request_sha256,request_bytes,evaluator_identity,state,evaluator_sandbox_id FROM requests WHERE request_id=?',
                             (ident,)).fetchone()
            identity_raw = canonical_bytes(envelope['evaluator_identity']).decode()
            if row is None:
                db.execute('INSERT INTO requests(request_id,request_sha256,request_bytes,evaluator_identity,evaluator_sandbox_id,state,created_at,updated_at) '
                           'VALUES(?,?,?,?,?,?,?,?)', (ident, envelope['request_sha256'], raw, identity_raw, envelope['evaluator_sandbox_id'], 'queued', now, now))
                state = 'queued'
            else:
                if row[0] != envelope['request_sha256'] or bytes(row[1]) != raw or row[2] != identity_raw or row[4] != envelope['evaluator_sandbox_id']:
                    raise ValueError('teacher request ID was rebound to different bytes or evaluator checkpoint')
                state = row[3]
            db.execute('COMMIT')
        return {'request_id': ident, 'request_sha256': envelope['request_sha256'], 'state': state}

    def claim(self, ident: str, owner: dict) -> tuple[str, bytes | None, dict | None]:
        if not ID_RE.fullmatch(ident):
            raise ValueError('teacher worker received an invalid request ID')
        now = time.time()
        with _connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT request_sha256,request_bytes,evaluator_identity,state,receipt_json,response_bytes,evaluator_sandbox_id '
                             'FROM requests WHERE request_id=?', (ident,)).fetchone()
            if row is None:
                raise ValueError('teacher worker has no durable request record')
            if row[3] in ('complete','failed','indeterminate'):
                db.execute('COMMIT')
                return row[3], bytes(row[1]) if row[3] == 'complete' else None, json.loads(row[4]) if row[4] else None
            if row[3] == 'submitted':
                prior = json.loads(row[4] or '{}')
                prior_owner = prior.get('worker_owner') or {}
                if _owned_process_alive(prior_owner):
                    db.execute('COMMIT')
                    return 'running', None, None
                receipt = self._receipt(ident, row[0], 'indeterminate', prior.get('teacher'), row[6], json.loads(row[2]),
                    response=b'', error={'type': 'IndeterminateSubmission',
                        'detail': 'worker ended after the durable submit marker; generation was not replayed'})
                db.execute('UPDATE requests SET state=?,response_sha256=?,response_bytes=?,response_status=?,receipt_json=?,updated_at=? '
                           'WHERE request_id=?', ('indeterminate', hashlib.sha256(b'').hexdigest(), b'', 502,
                            canonical_bytes(receipt).decode(), now, ident))
                db.execute('COMMIT')
                return 'indeterminate', None, receipt
            teacher = self.identity_provider()
            engine = teacher.get('active_engine') if isinstance(teacher, dict) else None
            processes = engine.get('processes') if isinstance(engine, dict) else None
            if (not isinstance(teacher, dict)
                    or teacher.get('sandbox_id') != TEACHER_SANDBOX_ID
                    or teacher.get('model_id') != TEACHER_MODEL_ID
            or teacher.get('serve_alias') != TEACHER_SERVE_ALIAS
                    or teacher.get('revision') != TEACHER_REVISION
                    or not isinstance(teacher.get('launch_identity'), dict)
                    or not isinstance(teacher.get('download_contract'), dict)
                    or teacher['download_contract'].get('repo') != TEACHER_MODEL_ID
                    or teacher['download_contract'].get('model_path') != '/work/generation/qwen36-35b-teacher-20261022/model'
                    or teacher['download_contract'].get('revision') != TEACHER_REVISION
                    or teacher['download_contract'].get('passed') is not True
                    or not isinstance(teacher['download_contract'].get('members_sha256'), str)
                    or not isinstance(teacher.get('source_bindings'), list)
                    or teacher.get('source_bindings') != list(TEACHER_SOURCE_BINDINGS)
                    or not isinstance(processes, list) or not processes):
                receipt = self._receipt(ident, row[0], 'failed', None, row[6], json.loads(row[2]),
                    response=b'', error={'type': 'TeacherBindingMismatch',
                        'detail': 'active engine/source identity does not match authorized teacher receipts',
                        'observed_teacher': teacher})
                db.execute('UPDATE requests SET state=?,response_sha256=?,response_bytes=?,response_status=?,receipt_json=?,updated_at=? '
                           'WHERE request_id=?', ('failed', hashlib.sha256(b'').hexdigest(), b'', 502,
                            canonical_bytes(receipt).decode(), now, ident))
                db.execute('COMMIT')
                return 'failed', None, receipt
            try:
                _validate_launch_process(teacher['launch_identity'], processes, _boot_id())
            except ValueError as error:
                receipt = self._receipt(ident, row[0], 'failed', None, row[6], json.loads(row[2]),
                    response=b'', error={'type': 'TeacherBindingMismatch', 'detail': str(error)})
                db.execute('UPDATE requests SET state=?,response_sha256=?,response_bytes=?,response_status=?,receipt_json=?,updated_at=? WHERE request_id=?',
                    ('failed', hashlib.sha256(b'').hexdigest(), b'', 502, canonical_bytes(receipt).decode(), now, ident))
                db.execute('COMMIT')
                return 'failed', None, receipt
            worker_owner = {**owner, 'boot_id': _boot_id()}
            evaluator_profile_for(row[6], json.loads(row[2]))
            submission = {'worker_owner': worker_owner, 'teacher': teacher, 'evaluator_sandbox_id': row[6]}
            db.execute('UPDATE requests SET state=?,receipt_json=?,updated_at=? WHERE request_id=?',
                       ('submitted', canonical_bytes(submission).decode(), now, ident))
            db.execute('COMMIT')
        return 'submitted', bytes(row[1]), teacher

    def _receipt(self, ident: str, request_sha: str, status: str, teacher: dict | None, evaluator_sandbox_id: str, evaluator_identity: dict, *,
                 response: bytes, response_status: int | None = None, error: dict | None = None) -> dict:
        return sign_receipt({'schema': RECEIPT_SCHEMA, 'request_id': ident, 'request_sha256': request_sha,
            'status': status, 'response_status': response_status,
            'response_sha256': hashlib.sha256(response).hexdigest(), 'response_bytes': len(response),
            'teacher': teacher or {'sandbox_id': TEACHER_SANDBOX_ID, 'model_id': TEACHER_MODEL_ID,
                'serve_alias': TEACHER_SERVE_ALIAS, 'revision': TEACHER_REVISION,
                'source_bindings': list(TEACHER_SOURCE_BINDINGS)},
            'evaluator_sandbox_id': evaluator_sandbox_id, 'evaluator_identity': evaluator_identity, 'created_at': time.time(), **({'error': error} if error else {})}, self.key)

    def finish(self, ident: str, response: bytes, response_status: int, before: dict, after: dict) -> dict:
        with _connect(self.path) as db:
            row = db.execute('SELECT request_sha256,state,receipt_json,evaluator_identity,evaluator_sandbox_id FROM requests WHERE request_id=?', (ident,)).fetchone()
        if row is None or row[1] != 'submitted':
            raise ValueError('teacher worker cannot finish an unsubmitted request')
        if before != after:
            return self.finish_indeterminate(ident, before, 'active teacher process/checkpoint binding changed during completion')
        if (response_status != 200 or len(response) > MAX_RESPONSE_BYTES or not _valid_chat_response(response)
                or json.loads(response).get('model') != TEACHER_SERVE_ALIAS):
            return self.finish_failed(ident, before, 'teacher returned a nonconforming chat-completions response')
        receipt = self._receipt(ident, row[0], 'complete', before, row[4], json.loads(row[3]), response=response, response_status=200)
        with _connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT state,request_sha256 FROM requests WHERE request_id=?', (ident,)).fetchone()
            if not current or current[0] != 'submitted' or current[1] != row[0]:
                raise ValueError('teacher result state/request binding changed before durable receipt')
            db.execute('UPDATE requests SET state=?,response_sha256=?,response_bytes=?,response_status=?,receipt_json=?,updated_at=? '
                       'WHERE request_id=?', ('complete', hashlib.sha256(response).hexdigest(), response, 200,
                        canonical_bytes(receipt).decode(), time.time(), ident))
            db.execute('COMMIT')
        return receipt

    def finish_failed(self, ident: str, teacher: dict | None, detail: str) -> dict:
        return self._finish_without_response(ident, 'failed', teacher, {'type': 'TeacherResponseRejected', 'detail': detail})

    def finish_indeterminate(self, ident: str, teacher: dict | None, detail: str) -> dict:
        return self._finish_without_response(ident, 'indeterminate', teacher, {'type': 'IndeterminateSubmission', 'detail': detail})

    def _finish_without_response(self, ident: str, status: str, teacher: dict | None, error: dict) -> dict:
        with _connect(self.path) as db:
            row = db.execute('SELECT request_sha256,state,evaluator_identity,evaluator_sandbox_id FROM requests WHERE request_id=?', (ident,)).fetchone()
        if row is None or row[1] != 'submitted':
            raise ValueError('teacher worker cannot settle a request that was not submitted')
        receipt = self._receipt(ident, row[0], status, teacher, row[3], json.loads(row[2]), response=b'', response_status=None, error=error)
        with _connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT state,request_sha256 FROM requests WHERE request_id=?', (ident,)).fetchone()
            if not current or current != (row[1], row[0]):
                raise ValueError('teacher request changed before signed terminal receipt')
            db.execute('UPDATE requests SET state=?,response_sha256=?,response_bytes=?,response_status=?,receipt_json=?,error_json=?,updated_at=? '
                       'WHERE request_id=?', (status, hashlib.sha256(b'').hexdigest(), b'', 502,
                        canonical_bytes(receipt).decode(), canonical_bytes(error).decode(), time.time(), ident))
            db.execute('COMMIT')
        return receipt

    def status(self, ident: str) -> dict | None:
        if not ID_RE.fullmatch(ident):
            raise ValueError('invalid teacher request ID')
        with _connect(self.path) as db:
            row = db.execute('SELECT request_sha256,state,response_bytes,response_status,receipt_json,error_json '
                             'FROM requests WHERE request_id=?', (ident,)).fetchone()
        if row is None:
            return None
        response = bytes(row[2]) if row[2] is not None else None
        return {'request_id': ident, 'request_sha256': row[0], 'state': row[1],
                'response_b64': base64.b64encode(response).decode() if response is not None else None,
                'response_sha256': hashlib.sha256(response).hexdigest() if response is not None else None,
                'response_status': row[3], 'receipt': json.loads(row[4]) if row[4] else None,
                'error': json.loads(row[5]) if row[5] else None}


def teacher_run(ident: str, store: TeacherJobStore, *, http_json=_http_json,
                process_reader=active_engine_processes):
    owner = {'pid': os.getpid(), 'start_ticks': _proc_start_ticks(os.getpid()), 'boot_id': _boot_id()}
    state, raw, teacher = store.claim(ident, owner)
    if state != 'submitted':
        return store.status(ident)
    before = teacher
    try:
        # The submit marker is already durable. Any transport exception from
        # here is ambiguous and must never cause a second POST for this ID.
        request_sha = store.status(ident)['request_sha256']
        with _connect(store.path) as db:
            request_row = db.execute('SELECT request_bytes FROM requests WHERE request_id=?', (ident,)).fetchone()
        if request_row is None or hashlib.sha256(bytes(request_row[0])).hexdigest() != request_sha:
            return store.finish_indeterminate(ident, before, 'durable teacher request bytes disappeared or changed')
        status, response = http_json('POST', '/v1/chat/completions', bytes(request_row[0]), timeout=900)
        after = store.identity_provider()
        store.finish(ident, response, status, before, after)
        return store.status(ident)
    except BaseException as error:
        # Once `submitted` is durable, even connection errors may mean the
        # generation reached vLLM. Preserve an honest terminal receipt.
        try:
            store.finish_indeterminate(ident, before,
                type(error).__name__ + ': submission may have reached the teacher; no replay was attempted')
            return store.status(ident)
        except Exception:
            raise


def _chunk_directory(root: Path, ident: str) -> Path:
    if not ID_RE.fullmatch(ident):
        raise ValueError('invalid chunked ferry request ID')
    return Path(root) / 'sdk-chunks' / ident


def write_chunked(root: Path, ident: str, raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) > MAX_TRANSFER_BYTES:
        raise ValueError('SDK ferry payload exceeds its hard transfer bound')
    directory = _chunk_directory(root, ident)
    count = max(1, (len(raw) + SDK_CHUNK_BYTES - 1) // SDK_CHUNK_BYTES)
    if count > (MAX_TRANSFER_BYTES + SDK_CHUNK_BYTES - 1) // SDK_CHUNK_BYTES:
        raise ValueError('SDK ferry chunk count exceeds bound')
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        part = raw[index * SDK_CHUNK_BYTES:(index + 1) * SDK_CHUNK_BYTES]
        chunk_path = directory / f'{index:06d}.bin'
        if not chunk_path.is_file() or chunk_path.read_bytes() != part:
            _durable_write(chunk_path, part)
    return {'request_id': ident, 'total_bytes': len(raw), 'chunk_count': count,
            'sha256': hashlib.sha256(raw).hexdigest()}


def read_chunked(root: Path, ident: str, total_bytes: int, chunk_count: int, digest: str) -> bytes:
    if (not ID_RE.fullmatch(ident) or type(total_bytes) is not int or not 0 <= total_bytes <= MAX_TRANSFER_BYTES
            or type(chunk_count) is not int or chunk_count != max(1, (total_bytes + SDK_CHUNK_BYTES - 1) // SDK_CHUNK_BYTES)
            or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('SDK ferry chunk manifest is outside the exact bounded format')
    directory = _chunk_directory(root, ident)
    parts = []
    for index in range(chunk_count):
        part = (directory / f'{index:06d}.bin').read_bytes()
        expected = SDK_CHUNK_BYTES if index < chunk_count - 1 else total_bytes - SDK_CHUNK_BYTES * (chunk_count - 1)
        if len(part) != expected:
            raise ValueError('SDK ferry chunk byte count differs from the exact manifest')
        parts.append(part)
    raw = b''.join(parts)
    if len(raw) != total_bytes or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('SDK ferry assembled bytes differ from the complete payload hash')
    return raw


def _durable_write(path: Path, raw: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _teacher_inbox_path(ident):
    if not ID_RE.fullmatch(ident):
        raise ValueError('invalid teacher request ID')
    return TEACHER_ROOT / 'inbox' / (ident + '.json')


def _teacher_result_path(ident):
    if not ID_RE.fullmatch(ident):
        raise ValueError('invalid teacher request ID')
    return TEACHER_ROOT / 'results' / (ident + '.json')


def _evaluator_result_path(ident):
    if not ID_RE.fullmatch(ident):
        raise ValueError('invalid evaluator request ID')
    return EVALUATOR_ROOT / 'inbox' / (ident + '.json')


def _launch_teacher_worker(ident: str):
    import fcntl
    directory = TEACHER_ROOT / 'workers'
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / (ident + '.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = TeacherJobStore()
        state = store.status(ident)
        if state is None:
            raise ValueError('teacher worker launch has no exact enqueued request')
        if state['state'] != 'queued':
            return state
        launch_path = directory / (ident + '.launch.json')
        if launch_path.is_file():
            try:
                old = json.loads(launch_path.read_bytes())
                if _owned_process_alive(old):
                    return {**state, 'worker': old}
            except (OSError, ValueError, TypeError):
                pass
        log_path = directory / (ident + '.log')
        with log_path.open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'teacher-run', ident],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                close_fds=True, cwd=str(TEACHER_ROOT))
        owner = {'pid': child.pid, 'start_ticks': _proc_start_ticks(child.pid), 'boot_id': _boot_id()}
        _durable_write(launch_path, canonical_bytes(owner))
        return {**state, 'worker': owner}


def _evaluator_queue(key_path=KEY_PATH, student='lfm'):
    profile = evaluator_profile(student)
    return EvaluatorQueue(EVALUATOR_DB, load_key(key_path), profile['native_identity'],
        identity_provider=lambda: evaluator_native_identity(student), evaluator_sandbox_id=profile['sandbox_id'])


def _teacher_store(key_path=KEY_PATH):
    return TeacherJobStore(TEACHER_DB, load_key(key_path))


def _publish_teacher_terminal(ident: str, value: dict):
    if (value.get('state') in ('complete','failed','indeterminate')
            and isinstance(value.get('receipt'), dict)
            and value['receipt'].get('schema') == RECEIPT_SCHEMA):
        raw = canonical_bytes({'request_id': ident,
            'response_b64': value.get('response_b64') or '', 'receipt': value['receipt']})
        return write_chunked(TEACHER_ROOT / 'results', ident, raw)
    return None


def _reconcile_teacher(ident: str, store: TeacherJobStore):
    value = store.status(ident)
    if value is None:
        raise ValueError('teacher status has no exact request')
    if value['state'] == 'submitted':
        stored = value.get('receipt') or {}
        owner = stored.get('worker_owner') or {}
        if not _owned_process_alive(owner):
            store.claim(ident, {'pid': os.getpid(), 'start_ticks': _proc_start_ticks(os.getpid()),
                                'boot_id': _boot_id()})
            value = store.status(ident)
    manifest = _publish_teacher_terminal(ident, value)
    result = {'request_id': ident, 'request_sha256': value['request_sha256'],
              'state': value['state']}
    if manifest is not None:
        result['result_manifest'] = manifest
    return result


def _assemble_sdk_chunks(root: Path, ident: str, total_bytes: int,
                         chunk_count: int, digest: str, destination: Path) -> dict:
    """Verify the complete SDK transfer before atomically installing its file."""
    raw = read_chunked(root, ident, total_bytes, chunk_count, digest)
    _durable_write(destination, raw)
    return {'request_id': ident, 'assembled_sha256': hashlib.sha256(raw).hexdigest(),
            'total_bytes': len(raw)}


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    serve = sub.add_parser('evaluator-serve')
    serve.add_argument('--student', choices=tuple(EVALUATOR_IDS), required=True)
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8766)
    serve.add_argument('--key-file', type=Path, default=KEY_PATH)
    next_job = sub.add_parser('evaluator-next')
    next_job.add_argument('--student', choices=tuple(EVALUATOR_IDS), required=True)
    next_job.add_argument('--key-file', type=Path, default=KEY_PATH)
    deliver = sub.add_parser('evaluator-deliver')
    deliver.add_argument('request_id')
    deliver.add_argument('--student', choices=tuple(EVALUATOR_IDS), required=True)
    deliver.add_argument('--key-file', type=Path, default=KEY_PATH)
    assemble_result = sub.add_parser('evaluator-assemble')
    assemble_result.add_argument('request_id')
    assemble_result.add_argument('--student', choices=tuple(EVALUATOR_IDS), required=True)
    assemble_result.add_argument('--bytes', type=int, required=True)
    assemble_result.add_argument('--chunks', type=int, required=True)
    assemble_result.add_argument('--sha256', required=True)
    enqueue = sub.add_parser('teacher-enqueue')
    enqueue.add_argument('request_id')
    enqueue.add_argument('--key-file', type=Path, default=KEY_PATH)
    assemble = sub.add_parser('teacher-assemble')
    assemble.add_argument('request_id')
    assemble.add_argument('--bytes', type=int, required=True)
    assemble.add_argument('--chunks', type=int, required=True)
    assemble.add_argument('--sha256', required=True)
    assemble.add_argument('--key-file', type=Path, default=KEY_PATH)
    status = sub.add_parser('teacher-status')
    status.add_argument('request_id')
    status.add_argument('--key-file', type=Path, default=KEY_PATH)
    worker = sub.add_parser('teacher-run')
    worker.add_argument('request_id')
    worker.add_argument('--key-file', type=Path, default=KEY_PATH)
    args = parser.parse_args(argv)
    if args.operation == 'evaluator-serve':
        profile = evaluator_profile(args.student)
        server = evaluator_server(path=EVALUATOR_DB, key_path=args.key_file,
                                  host=args.host, port=args.port, student=args.student)
        print(json.dumps({'host': args.host, 'port': args.port, 'database': str(EVALUATOR_DB),
                          'student': args.student, 'sandbox_id': profile['sandbox_id'],
                          'identity': profile['native_identity']}, separators=(',', ':')), flush=True)
        try:
            server.serve_forever(poll_interval=.25)
        finally:
            server.server_close()
        return 0
    if args.operation == 'evaluator-next':
        print(json.dumps(_evaluator_queue(args.key_file, args.student).next_manifest(), separators=(',', ':')))
        return 0
    if args.operation == 'evaluator-assemble':
        evaluator_profile(args.student)
        if not ID_RE.fullmatch(args.request_id):
            raise ValueError('invalid evaluator result request ID')
        assembled = _assemble_sdk_chunks(EVALUATOR_ROOT / 'inbox',
            args.request_id, args.bytes, args.chunks, args.sha256,
            _evaluator_result_path(args.request_id))
        value = json.loads(_evaluator_result_path(args.request_id).read_bytes())
        if (not isinstance(value, dict) or set(value) != {'request_id', 'response_b64', 'receipt'}
                or value['request_id'] != args.request_id):
            _evaluator_result_path(args.request_id).unlink(missing_ok=True)
            raise ValueError('assembled evaluator result has an unknown or rebound envelope')
        print(json.dumps(assembled, separators=(',', ':')))
        return 0
    if args.operation == 'evaluator-deliver':
        path = _evaluator_result_path(args.request_id)
        if not path.is_file():
            raise ValueError('exact evaluator ferry receipt file is missing')
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict) or set(value) != {'request_id','response_b64','receipt'} or value['request_id'] != args.request_id:
            raise ValueError('evaluator ferry receipt file changed its exact request binding')
        response = base64.b64decode(value['response_b64'], validate=True)
        print(json.dumps(_evaluator_queue(args.key_file, args.student).complete(args.request_id, response, value['receipt']),
                         separators=(',', ':')))
        return 0
    if args.operation == 'teacher-assemble':
        if not ID_RE.fullmatch(args.request_id):
            raise ValueError('invalid teacher request ID')
        _assemble_sdk_chunks(TEACHER_ROOT / 'inbox',
            args.request_id, args.bytes, args.chunks, args.sha256,
            _teacher_inbox_path(args.request_id))
        value = json.loads(_teacher_inbox_path(args.request_id).read_bytes())
        result = _teacher_store(args.key_file).enqueue(value)
        if result['state'] == 'queued':
            result = {**result, 'worker': _launch_teacher_worker(args.request_id)}
        else:
            result = _reconcile_teacher(args.request_id, _teacher_store(args.key_file))
            if result['state'] == 'queued':
                result = {**result, 'worker': _launch_teacher_worker(args.request_id)}
        print(json.dumps({key: result[key] for key in ('request_id', 'request_sha256', 'state')},
                         separators=(',', ':')))
        return 0
    if args.operation == 'teacher-enqueue':
        path = _teacher_inbox_path(args.request_id)
        if not path.is_file():
            raise ValueError('exact teacher ferry request file is missing')
        value = json.loads(path.read_bytes())
        result = _teacher_store(args.key_file).enqueue(value)
        if result['state'] == 'queued':
            result = {**result, 'worker': _launch_teacher_worker(args.request_id)}
        else:
            result = _reconcile_teacher(args.request_id, _teacher_store(args.key_file))
            if result['state'] == 'queued':
                result = {**result, 'worker': _launch_teacher_worker(args.request_id)}
        print(json.dumps(result, separators=(',', ':')))
        return 0
    if args.operation == 'teacher-status':
        store = _teacher_store(args.key_file)
        value = store.status(args.request_id)
        if value is None:
            value = {'request_id': args.request_id, 'state': 'missing'}
        else:
            value = _reconcile_teacher(args.request_id, store)
        print(json.dumps(value, separators=(',', ':')))
        return 0
    if args.operation == 'teacher-run':
        value = teacher_run(args.request_id, _teacher_store(args.key_file))
        _publish_teacher_terminal(args.request_id, value or {})
        return 0
    parser.error('unknown ferry operation')


if __name__ == '__main__':
    raise SystemExit(cli())
