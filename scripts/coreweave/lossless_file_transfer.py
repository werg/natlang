"""Lossless, identity-bound streaming byte differences for immutable artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import uuid

MAGIC = b'NATLANG-XOR-ZSTD\x00\x01'
SCHEMA = 'natlang.lossless-byte-transfer/1'
BLOCK_BYTES = 8 << 20
MAX_HEADER_BYTES = 64 << 10


def _identity(value):
    if (not isinstance(value, dict) or set(value) != {'sha256', 'bytes'}
            or not isinstance(value.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', value['sha256'])
            or type(value.get('bytes')) is not int or value['bytes'] < 0):
        raise ValueError('lossless transfer requires exact content identities')
    return value


def _stamp(path, identity):
    path = Path(path).resolve(strict=True)
    value = path.stat()
    if not stat.S_ISREG(value.st_mode) or value.st_size != identity['bytes']:
        raise ValueError('lossless input size or file kind differs')
    return path, (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _unchanged(path, before):
    value = path.stat()
    if before != (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns):
        raise ValueError('lossless input changed during processing')


def _block_size(value):
    if type(value) is not int or not 0 < value <= 64 << 20:
        raise ValueError('lossless block size exceeds bounded decoder memory')
    return value


def _xor(left, right):
    import numpy as np
    if len(right) < len(left):
        right += b'\0' * (len(left) - len(right))
    return np.bitwise_xor(np.frombuffer(left, dtype=np.uint8),
                         np.frombuffer(right, dtype=np.uint8)).tobytes()


class _HashStream:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data

    def write(self, data):
        count = self.stream.write(data)
        if count != len(data):
            raise OSError('short lossless artifact write')
        self.digest.update(data)
        self.size += count
        return count

    def flush(self):
        self.stream.flush()

    def identity(self):
        return {'sha256': self.digest.hexdigest(), 'bytes': self.size}


def _output(path, inputs):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if any(path.resolve() == value for value in inputs):
        raise ValueError('lossless output aliases an input')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.with_name('.' + path.name + '.pending-' + uuid.uuid4().hex)


def _publish(pending, destination):
    # The finished, synced artifact is immutable before its CAS publication.
    os.chmod(pending, 0o444)
    os.link(pending, destination)
    pending.unlink()
    descriptor = os.open(destination.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def encode(target, basis, destination, *, target_identity, basis_identity,
           block_bytes=BLOCK_BYTES, compression_level=3, progress=None):
    import zstandard as zstd
    target_identity, basis_identity = _identity(target_identity), _identity(basis_identity)
    block_bytes = _block_size(block_bytes)
    target, target_stamp = _stamp(target, target_identity)
    basis, basis_stamp = _stamp(basis, basis_identity)
    destination, pending = _output(destination, (target, basis))
    header = {'schema': SCHEMA, 'target': target_identity, 'basis': basis_identity,
              'codec': 'xor-zstd', 'block_bytes': block_bytes}
    raw = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    try:
        with target.open('rb') as source, basis.open('rb') as reference, pending.open('xb') as file:
            original, base, output = _HashStream(source), _HashStream(reference), _HashStream(file)
            output.write(MAGIC + struct.pack('>I', len(raw)) + raw)
            with zstd.ZstdCompressor(level=compression_level).stream_writer(output, closefd=False) as compressor:
                while chunk := original.read(block_bytes):
                    compressor.write(_xor(chunk, base.read(len(chunk))))
                    if progress:
                        progress({'target_bytes': original.size, 'basis_bytes': base.size,
                                  'container_bytes': output.size})
            while base.read(block_bytes):
                pass
            if original.identity() != target_identity or base.identity() != basis_identity:
                raise ValueError('lossless source or basis content differs')
            _unchanged(target, target_stamp)
            _unchanged(basis, basis_stamp)
            output.flush()
            os.fsync(file.fileno())
            container_identity = output.identity()
        _publish(pending, destination)
        return {'path': str(destination), 'identity': container_identity, 'manifest': header}
    finally:
        pending.unlink(missing_ok=True)


def _header(source):
    if source.read(len(MAGIC)) != MAGIC:
        raise ValueError('unknown lossless artifact encoding')
    raw_size = source.read(4)
    if len(raw_size) != 4:
        raise ValueError('truncated lossless header length')
    size = struct.unpack('>I', raw_size)[0]
    if not 0 < size <= MAX_HEADER_BYTES:
        raise ValueError('lossless header size is invalid')
    raw = source.read(size)
    if len(raw) != size:
        raise ValueError('truncated lossless header')
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schema', 'target', 'basis', 'codec', 'block_bytes'}
            or value['schema'] != SCHEMA or value['codec'] != 'xor-zstd'):
        raise ValueError('lossless manifest schema differs')
    _identity(value['target'])
    _identity(value['basis'])
    _block_size(value['block_bytes'])
    return value


def _restore(container, basis, *, container_identity, target_identity, basis_identity,
             sink=None, block_bytes=BLOCK_BYTES, progress=None):
    import zstandard as zstd
    container_identity = _identity(container_identity)
    target_identity, basis_identity = _identity(target_identity), _identity(basis_identity)
    block_bytes = _block_size(block_bytes)
    container, container_stamp = _stamp(container, container_identity)
    basis, basis_stamp = _stamp(basis, basis_identity)
    target_hash, target_size = hashlib.sha256(), 0
    with container.open('rb') as encoded, basis.open('rb') as reference:
        source, base = _HashStream(encoded), _HashStream(reference)
        header = _header(source)
        if header['target'] != target_identity or header['basis'] != basis_identity:
            raise ValueError('lossless manifest does not bind the selected target and basis')
        with zstd.ZstdDecompressor(max_window_size=64 << 20).stream_reader(
                source, read_across_frames=True, closefd=False) as decoder:
            while target_size < target_identity['bytes']:
                chunk = decoder.read(min(block_bytes, target_identity['bytes'] - target_size))
                if not chunk:
                    raise ValueError('lossless target stream is incomplete')
                restored = _xor(chunk, base.read(len(chunk)))
                target_hash.update(restored)
                target_size += len(restored)
                if sink is not None:
                    if sink.write(restored) != len(restored):
                        raise OSError('short restored artifact write')
                if progress:
                    progress({'target_bytes': target_size, 'basis_bytes': base.size,
                              'container_bytes': source.size})
            if decoder.read(1):
                raise ValueError('lossless stream exceeds its declared target')
        while base.read(block_bytes):
            pass
        while source.read(block_bytes):
            pass
        if (source.identity() != container_identity or base.identity() != basis_identity
                or {'sha256': target_hash.hexdigest(), 'bytes': target_size} != target_identity):
            raise ValueError('lossless container, basis or restored target identity differs')
        _unchanged(container, container_stamp)
        _unchanged(basis, basis_stamp)
    return {'identity': target_identity, 'manifest': header, 'exact_reconstruction': True}


def verify(container, basis, **kwargs):
    """Verify the entire reconstruction without allocating a second full file."""
    return _restore(container, basis, **kwargs)


def decode(container, basis, destination, **kwargs):
    """Publish only a fully authenticated, byte-identical reconstructed artifact."""
    inputs = tuple(Path(path).resolve(strict=True) for path in (container, basis))
    destination, pending = _output(destination, inputs)
    try:
        with pending.open('xb') as stream:
            result = _restore(container, basis, sink=stream, **kwargs)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(pending, destination)
        return {**result, 'path': str(destination)}
    finally:
        pending.unlink(missing_ok=True)
