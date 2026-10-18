"""Durable checkpoints for complete, per-model acquired-admission checks.

This cache can skip a model verifier only after that verifier completed and
only while every declared input, current policy binding, and verifier module
still has the same exact identity. It never grants training admission.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from natlang_neuralese.common.hashing import (
    canonical_json_sha256_hex,
    sha256_file_hex,
    sha256_file_hex_cached,
)
from natlang_neuralese.common.jsonio import canonical_json_bytes


SCHEMA = 'natlang.acquired-admission-model-checkpoint/1'


def artifact_bindings(paths):
    """Bind regular artifacts to bytes through the race-checked strong cache."""
    rows = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        if path in seen:
            continue
        seen.add(path)
        status = path.stat()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'admission checkpoint input is not a regular file: {path}')
        digest = sha256_file_hex_cached(path, trust_archive_receipt=False)
        after = path.stat()
        before_identity = (status.st_dev, status.st_ino, status.st_size,
                           status.st_mtime_ns, status.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity:
            raise RuntimeError(f'admission checkpoint input changed during binding: {path}')
        rows.append({'path': str(path), 'sha256': digest, 'bytes': status.st_size})
    return sorted(rows, key=lambda row: row['path'])


def verifier_bindings(paths):
    """Bind code bytes uncached so a continuation cannot inherit an old policy."""
    rows = []
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        rows.append({'path': str(path), 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size})
    return sorted(rows, key=lambda row: row['path'])


def model_checkpoint_identity(*, source, model, policy, artifacts, verifier_files):
    """Build the exact identity for one fully verified model/source pair."""
    return {
        'schema': SCHEMA,
        'source': source,
        'model': model,
        'policy': policy,
        'artifacts': artifact_bindings(artifacts),
        'verifier_files': verifier_bindings(verifier_files),
    }


def _checkpoint_paths(root: Path, identity):
    key = canonical_json_sha256_hex(identity)
    return key, sorted(root.glob(f'{key}-*.json'))


def load_model_checkpoint(root, identity):
    """Return a complete cached verifier result or None on an identity miss."""
    root = Path(root)
    key, candidates = _checkpoint_paths(root, identity)
    identity_bytes = canonical_json_bytes(identity)
    if len(candidates) > 1:
        raise ValueError('multiple acquired admission results exist for one exact model identity')
    for path in candidates:
        payload = json.loads(path.read_text(encoding='utf-8'))
        # Identities are content-addressed as JSON.  In-memory tuples (for
        # example decision_contract rows) serialize as arrays and reload as
        # lists, so Python container equality would miss the exact checkpoint
        # and needlessly recompute completed shards.
        if (payload.get('schema') != SCHEMA
                or canonical_json_bytes(payload.get('identity')) != identity_bytes):
            continue
        result = payload.get('result')
        result_sha = canonical_json_sha256_hex(result)
        if path.name != f'{key}-{result_sha}.json':
            raise ValueError('acquired admission checkpoint content identity mismatch')
        return result
    return None


def save_model_checkpoint(root, identity, result):
    """Atomically publish one immutable completed model-verification result."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = canonical_json_sha256_hex(identity)
    result_sha = canonical_json_sha256_hex(result)
    destination = root / f'{key}-{result_sha}.json'
    existing_candidates = list(root.glob(f'{key}-*.json'))
    if any(path != destination for path in existing_candidates):
        raise ValueError('conflicting acquired admission results exist for one exact model identity')
    if destination.exists():
        payload = json.loads(destination.read_text(encoding='utf-8'))
        if (canonical_json_bytes(payload.get('identity')) != canonical_json_bytes(identity)
                or canonical_json_bytes(payload.get('result')) != canonical_json_bytes(result)):
            raise ValueError('immutable acquired admission checkpoint path has different content')
        return destination

    payload = {'schema': SCHEMA, 'identity': identity, 'result': result}
    data = canonical_json_bytes(payload) + b'\n'
    fd, temp_name = tempfile.mkstemp(prefix='.admission-checkpoint-', suffix='.tmp', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_name, destination)
        except FileExistsError:
            payload_existing = json.loads(destination.read_text(encoding='utf-8'))
            if payload_existing.get('identity') != identity or payload_existing.get('result') != result:
                raise ValueError('concurrent admission checkpoint publisher wrote different content')
        directory_fd = os.open(root, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return destination


def load_or_compute_model_checkpoint(root, identity, compute):
    """Use only a complete exact-key result; otherwise run and seal the verifier."""
    cached = load_model_checkpoint(root, identity)
    if cached is not None:
        key, candidates = _checkpoint_paths(Path(root), identity)
        return cached, candidates[0], True
    result = compute()
    path = save_model_checkpoint(root, identity, result)
    return result, path, False
