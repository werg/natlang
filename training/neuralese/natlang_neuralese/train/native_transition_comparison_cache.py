"""Local authenticated memo of an independently verified transition comparison.

A memo is only an acceleration hint. Callers must freshly authenticate both
checkpoint files and an independently bound proof before looking it up.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from ..common.jsonio import canonical_json_bytes

SCHEMA = 'natlang.native-transition-comparison-cache/1'
MAX_ENTRIES = 128


def _code_identity(code):
    from types import CodeType
    def constant(value):
        if isinstance(value, CodeType): return {'code': _code_identity(value)}
        if isinstance(value, tuple): return {'tuple': [constant(item) for item in value]}
        if isinstance(value, frozenset): return {'frozenset': sorted(map(repr, value))}
        return {'type': type(value).__qualname__, 'value': repr(value)}
    return canonical_json_sha256_hex({'bytecode': code.co_code.hex(),
        'constants': [constant(item) for item in code.co_consts], 'names': code.co_names,
        'variables': code.co_varnames, 'freevars': code.co_freevars, 'cellvars': code.co_cellvars,
        'argcount': code.co_argcount, 'posonly': code.co_posonlyargcount,
        'kwonly': code.co_kwonlyargcount, 'flags': code.co_flags,
        'exceptions': code.co_exceptiontable.hex()})


def _callable_identity(value):
    code = getattr(value, '__code__', None)
    return {'module': getattr(value, '__module__', None),
            'name': getattr(value, '__qualname__', type(value).__qualname__),
            'code': None if code is None else _code_identity(code),
            'defaults': repr(getattr(value, '__defaults__', None)),
            # A closure may carry mutable policy. Bind its live identity; such
            # customized callables intentionally cannot share across processes.
            'closure': [id(cell.cell_contents) for cell in (getattr(value, '__closure__', None) or ())]}


def verifier_identity():
    """Bind live comparison callables, their source closure, and runtime binaries."""
    import numpy
    import torch
    from ..common import hashing, jsonio
    from ..common.checkpoint_digest import verified_checkpoint_digest
    from ..model import native_model, native_parameter_precision, shared_execution
    from ..rl import native_direct_sft_recovery, native_raw_policy
    from . import native_continuation, native_execution_transition, native_zero_sgd_adoption
    modules = (sys.modules[__name__], hashing, jsonio, native_model, native_parameter_precision,
               shared_execution, native_direct_sft_recovery, native_raw_policy,
               native_continuation, native_execution_transition, native_zero_sgd_adoption)
    files = []
    for module in modules:
        path = Path(module.__file__).resolve(strict=True)
        before = path.stat()
        digest = sha256_file_hex(path)
        after = path.stat()
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if signature(before) != signature(after):
            raise RuntimeError('transition verifier source changed while hashing')
        files.append({'path': str(path), 'sha256': digest})
    functions = (native_execution_transition.derive, native_execution_transition.verify_states,
                 native_zero_sgd_adoption._same, native_raw_policy.require_raw_training,
                 native_raw_policy.saved_native_identity, native_model.native_tensor_identity,
                 native_direct_sft_recovery._verify_historical_file_ref,
                 native_direct_sft_recovery._strong_binding, hashing.sha256_file_hex,
                 native_parameter_precision.ordinary_precision_state_dtypes,
                 native_parameter_precision.validate_norm_precision,
                 native_parameter_precision.validate_norm_execution,
                 shared_execution.validate_shared_execution_profile, torch.load, torch.equal,
                 numpy.array_equal)
    binaries = []
    for path in (Path(torch._C.__file__), Path(torch.__file__).parent/'lib/libtorch_cpu.so',
                 Path(numpy._core._multiarray_umath.__file__)):
        path = path.resolve(strict=True)
        st = path.stat()
        binaries.append({'path': str(path), 'identity': [st.st_dev, st.st_ino, st.st_size,
                                                         st.st_mtime_ns, st.st_ctime_ns],
                         'sha256': verified_checkpoint_digest(path)})
    basis = {'files': files, 'callables': [_callable_identity(f) for f in functions],
        'python': sys.version, 'torch': str(torch.__version__), 'numpy': numpy.__version__,
        'binaries': binaries,
        'controls': {'transition_schema': native_execution_transition.SCHEMA,
                     'source': native_execution_transition.SOURCE,
                     'target': native_execution_transition.TARGET,
                     'checkpoint_schema': native_model.CHECKPOINT_SCHEMA,
                     'training_schema': native_continuation.TRAINING_SCHEMA,
                     'default_dtype': str(torch.get_default_dtype()),
                     'force_weights_only': os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD'),
                     'force_no_weights_only': os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')}}
    return canonical_json_sha256_hex(basis)


def _directory():
    from ..common.hashing import _hash_cache_path
    cache = _hash_cache_path()
    return Path(os.environ.get('NATLANG_EXECUTION_TRANSITION_CACHE') or
                cache.with_name(cache.name + '.native-transition-comparisons-v1'))


def _private(path, *, directory=False):
    st = path.lstat()
    return (st.st_uid == os.geteuid() and not st.st_mode & 0o077 and
            (stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)))


def _read_secret(directory):
    path = directory/'key'
    if not _private(directory, directory=True) or not _private(path):
        raise ValueError('transition comparison cache is not private')
    secret = path.read_bytes()
    if len(secret) != 32: raise ValueError('invalid transition comparison cache key')
    return secret


def _mac(secret, body):
    return hmac.new(secret, canonical_json_bytes(body), hashlib.sha256).hexdigest()


def lookup(key, expected):
    """Return only the already authenticated independent proof or a cache miss."""
    directory = _directory()
    try:
        secret = _read_secret(directory)
        path = directory/(canonical_json_sha256_hex(key)+'.json')
        if not _private(path) or path.stat().st_size > 1024*1024: return None
        before = path.stat(); raw = path.read_bytes(); after = path.stat()
        sig = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if sig(before) != sig(after): return None
        record = json.loads(raw)
        body = record['body']
        if (set(record) != {'body','mac'} or not isinstance(record['mac'], str)
                or not hmac.compare_digest(record['mac'], _mac(secret, body))
                or canonical_json_bytes(body) != canonical_json_bytes(
                    {'schema': SCHEMA, 'key': key, 'result': expected,
                     'result_sha256': canonical_json_sha256_hex(expected)})):
            return None
        return expected
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def store(key, result):
    """Publish only after the caller's full comparison and independent proof match."""
    import fcntl
    directory = _directory()
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not _private(directory, directory=True): return
        lock_fd = os.open(directory/'lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            secret_path = directory/'key'
            if not secret_path.exists():
                fd = os.open(secret_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(os.urandom(32));handle.flush();os.fsync(handle.fileno())
            secret = _read_secret(directory)
            body = {'schema': SCHEMA, 'key': key, 'result': result,
                    'result_sha256': canonical_json_sha256_hex(result)}
            record = {'body': body, 'mac': _mac(secret, body)}
            fd, pending = tempfile.mkstemp(prefix='.comparison-', dir=directory)
            try:
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(canonical_json_bytes(record));handle.flush();os.fsync(handle.fileno())
                target = directory/(canonical_json_sha256_hex(key)+'.json')
                os.replace(pending, target)
                records = sorted((path for path in directory.glob('*.json') if path != target),
                                 key=lambda path: path.stat().st_mtime_ns, reverse=True)
                for stale in records[MAX_ENTRIES-1:]: stale.unlink(missing_ok=True)
                directory_fd = os.open(directory, os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(directory_fd)
                finally: os.close(directory_fd)
            finally:
                Path(pending).unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        # The optimization never makes a successful verification depend on cache I/O.
        return
