"""Hash chained, durable transaction storage for a fenced DiLoCo replica.

This module only provides storage. It does not authorize promotion or certify a
coordinator operation until the caller has also verified the state-machine replay.
"""
from __future__ import annotations
import hashlib
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

from ...common.jsonio import write_json_durable

SCHEMA = 'natlang.diloco-replica-transaction/1'
CHUNK = 1 << 20


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(CHUNK): h.update(block)
    return h.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ReplicaJournal:
    """Single-writer ordered transactions, published only after durable payloads."""
    def __init__(self, root, *, run_identity, baseline_sha256, max_payload_bytes=8 << 30, max_transactions=20000):
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:raise ValueError('invalid replica journal budget')
        if type(max_transactions) is not int or max_transactions <= 0:raise ValueError('invalid replica journal transaction budget')
        self.max_payload_bytes=max_payload_bytes;self.max_transactions=max_transactions
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / '.writer.lock').open('a+b')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise ValueError('replica journal already has a writer') from None
        self.identity = {'run_identity': run_identity, 'baseline_sha256': baseline_sha256}
        p = self.root / 'identity.json'
        if p.exists():
            if json.loads(p.read_text()) != self.identity: raise ValueError('replica baseline or run differs')
        else:
            write_json_durable(p, self.identity)
        self.verify()

    def close(self):
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def verify(self, *, expected_sequence=None, expected_head=None):
        paths = sorted(self.root.glob('[0-9]' * 20))
        previous = _identity(self.identity)
        total = 0
        for number, directory in enumerate(paths, 1):
            if directory.name != f'{number:020d}' or not directory.is_dir(): raise ValueError('replica journal gap')
            value = json.loads((directory/'record.json').read_text())
            if value.get('schema') != SCHEMA or value.get('sequence') != number or value.get('previous') != previous:
                raise ValueError('replica journal chain mismatch')
            payload = directory/'payload'
            if not payload.is_file() or payload.stat().st_size != value.get('payload_bytes') or _digest(payload) != value.get('payload_sha256'):
                raise ValueError('replica journal payload mismatch')
            total += value['payload_bytes']
            previous = _identity(value)
        if total > self.max_payload_bytes or len(paths) > self.max_transactions:raise ValueError('replica journal exceeds storage budget')
        self.payload_bytes=total
        if expected_sequence is not None and (len(paths) != expected_sequence or previous != expected_head):
            raise ValueError('replica journal differs from authoritative committed head')
        self.sequence = len(paths)
        self.head = previous
        return {'sequence': self.sequence, 'head': self.head}

    def append(self, event, payload, *, expected_sequence, expected_previous):
        source = Path(payload)
        if expected_sequence == self.sequence and expected_sequence > 0:
            record = json.loads((self.root/f'{expected_sequence:020d}'/'record.json').read_text())
            if (record['previous'] != expected_previous or record['event'] != event
                    or record['payload_bytes'] != source.stat().st_size or record['payload_sha256'] != _digest(source)):
                raise ValueError('replica journal duplicate differs')
            return {'sequence': self.sequence, 'head': self.head}
        if expected_sequence != self.sequence + 1 or expected_previous != self.head:
            raise ValueError('replica journal sequence or prior head differs')
        if len(json.dumps(event,sort_keys=True,separators=(',',':')).encode()) > 65536:
            raise ValueError('replica journal event exceeds metadata budget')
        if self.sequence >= self.max_transactions or self.payload_bytes + source.stat().st_size > self.max_payload_bytes:
            raise ValueError('replica journal budget exhausted; compact a verified generation')
        private = Path(tempfile.mkdtemp(prefix='.pending-', dir=self.root))
        try:
            target = private/'payload'; h = hashlib.sha256(); total = 0
            with source.open('rb') as src, target.open('xb') as dst:
                while block := src.read(CHUNK):
                    dst.write(block); h.update(block); total += len(block)
                dst.flush(); os.fsync(dst.fileno())
            record = {'schema': SCHEMA, 'sequence': expected_sequence, 'previous': expected_previous,
                      'event': event, 'payload_sha256': h.hexdigest(), 'payload_bytes': total}
            write_json_durable(private/'record.json', record); _sync_dir(private)
            destination = self.root/f'{expected_sequence:020d}'
            if destination.exists(): raise ValueError('replica journal sequence already exists')
            os.rename(private, destination); _sync_dir(self.root)
            self.sequence = expected_sequence; self.head = _identity(record); self.payload_bytes += total
            return {'sequence': self.sequence, 'head': self.head}
        finally:
            if private.exists(): shutil.rmtree(private)
