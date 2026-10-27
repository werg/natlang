"""Durable owner-scoped storage for native positional state and named tensors."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import sqlite3
import struct
import threading
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

# The exact tensor inventory comes from NativeStateBlock.tensors().
NATIVE_HEADER = "natlang.native-state/1"
_NATIVE_DTYPES = {torch.float16: ("F16", 2), torch.bfloat16: ("BF16", 2),
                  torch.float32: ("F32", 4), torch.float64: ("F64", 8), torch.int64: ("I64", 8)}
_NATIVE_FROM_SAFE = {value[0]: (key, value[1]) for key, value in _NATIVE_DTYPES.items()}


def _native_tensor_bytes(tensor: torch.Tensor) -> bytes:
    data = tensor.detach().contiguous().cpu()
    if data.dtype == torch.bfloat16:
        return data.view(torch.int16).numpy().tobytes()
    return data.numpy().tobytes()


def native_state_id(block) -> str:
    """Cross-language ID over canonical provenance, ordered descriptors and every tensor byte."""
    metadata = json.dumps(block.metadata(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    digest.update(NATIVE_HEADER.encode() + b"\0" + struct.pack("<Q", len(metadata)) + metadata)
    for name, tensor in sorted(block.tensors().items()):
        dtype = _NATIVE_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"unsupported native state dtype {tensor.dtype}")
        encoded_name = name.encode()
        raw = _native_tensor_bytes(tensor)
        descriptor = json.dumps({"dtype": dtype[0], "shape": list(tensor.shape)}, separators=(",", ":")).encode()
        digest.update(struct.pack("<I", len(encoded_name)) + encoded_name)
        digest.update(struct.pack("<I", len(descriptor)) + descriptor)
        digest.update(struct.pack("<Q", len(raw)) + raw)
    return "nzs1_" + base64.b32encode(digest.digest()).decode().rstrip("=").lower()


@dataclass
class NativeTensorRecord:
    """Named native parameter/gradient/update artifact in the shared store."""
    id: str
    artifact: object

    def meta(self):
        return {'id': self.id, **self.artifact.metadata()}


def encode_native_state(block) -> bytes:
    """Encode a validated NativeStateBlock as deterministic multi-tensor safetensors."""
    from ..model.state_block import NativeStateBlock
    if not isinstance(block, NativeStateBlock):
        raise TypeError("expected NativeStateBlock")
    block.validate()
    block_id = native_state_id(block)
    entries, payloads, offset = {}, [], 0
    for name, tensor in sorted(block.tensors().items()):
        safe_dtype, _ = _NATIVE_DTYPES[tensor.dtype]
        raw = _native_tensor_bytes(tensor)
        entries[name] = {"dtype": safe_dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        payloads.append(raw)
        offset += len(raw)
    metadata = {"id": block_id, **block.metadata()}
    entries["__metadata__"] = {"natlang.native_state": json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
    header = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + b"".join(payloads)


def decode_native_state(body: bytes):
    """Decode and verify a native state artifact; reject gaps, overlaps, unknown tensors and false IDs."""
    from ..model.state_block import NativeStateBlock
    try:
        if len(body) < 8:
            raise ValueError("not a safetensors body")
        (header_size,) = struct.unpack("<Q", body[:8])
        if header_size > len(body) - 8:
            raise ValueError("header exceeds the body")
        header = json.loads(body[8:8 + header_size])
        if not isinstance(header, dict):
            raise ValueError("native state header must be an object")
        metadata = json.loads(header.get("__metadata__", {}).get("natlang.native_state", "{}"))
        if not isinstance(metadata, dict):
            raise ValueError("native state metadata must be an object")
        tensors, spans = {}, []
        payload = body[8 + header_size:]
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype_info = _NATIVE_FROM_SAFE.get(entry["dtype"])
            if dtype_info is None:
                raise ValueError(f"unsupported native state dtype {entry['dtype']}")
            dtype, itemsize = dtype_info
            shape, offsets = entry["shape"], entry["data_offsets"]
            if (not isinstance(shape, list) or not shape or any(type(dim) is not int or dim < 0 for dim in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets)):
                raise ValueError(f"invalid native tensor descriptor: {name}")
            start, end = offsets
            count = 1
            for dim in shape:
                count *= dim
            if start < 0 or end < start or end > len(payload) or end - start != count * itemsize:
                raise ValueError(f"invalid native tensor byte range: {name}")
            spans.append((start, end, name))
            raw = payload[start:end]
            torch_dtype = dtype
            tensor = (torch.empty(0, dtype=dtype) if not raw else torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16) if dtype == torch.bfloat16
                      else torch.frombuffer(bytearray(raw), dtype=dtype))
            tensors[name] = tensor.reshape(shape).clone()
        spans.sort()
        if not spans or spans[0][0] != 0 or spans[-1][1] != len(payload) or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("native state tensor payload has gaps or overlaps")
        block = NativeStateBlock.from_tensors(metadata, tensors)
        actual = native_state_id(block)
        if metadata.get("id") != actual:
            raise ValueError(f"native state ID does not match its content ({actual})")
        return block, actual
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed native state: {error}") from error



@dataclass
class NativeStateRecord:
    """Archived native positional state."""
    id: str
    block: object

    def meta(self) -> dict:
        return {"id": self.id, **self.block.metadata()}



class TensorStore:
    """Content-addressed native artifact store with owner-scoped holds and pins, optionally spilled to disk.

    Owners are sessions or runtimes (`x-natlang-owner` on the wire). They hold native state and named tensor
    artifacts they upload, receive, pin or list as referenced. Collection releases only the requesting owner's
    unpinned artifacts. A directory enables durable safetensors storage and a bounded resident LRU.
    """

    def __init__(self, directory: str | os.PathLike | None = None, resident_bytes: int | None = None):
        self.directory = Path(directory) if directory is not None else None
        self.resident_bytes = resident_bytes
        self._resident: OrderedDict[str, NativeStateRecord | NativeTensorRecord] = OrderedDict()
        self._resident_size = 0
        self._lock = threading.Lock()
        if self.directory is not None:
            (self.directory / "blocks").mkdir(parents=True, exist_ok=True)
            self._lockfile = open(self.directory / "store.lock", "a")
            try:
                fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lockfile.close()
                raise RuntimeError(f"another process owns the block store at {self.directory}") from None
        self._db = sqlite3.connect(":memory:" if self.directory is None else str(self.directory / "store.sqlite"),
                                   check_same_thread=False, isolation_level=None)
        self._db.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE TABLE IF NOT EXISTS blocks (id TEXT PRIMARY KEY, bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS holds (id TEXT NOT NULL, owner TEXT NOT NULL, PRIMARY KEY (id, owner));
            CREATE INDEX IF NOT EXISTS holds_by_owner ON holds (owner);
            CREATE TABLE IF NOT EXISTS pins (id TEXT NOT NULL, owner TEXT NOT NULL, count INTEGER NOT NULL,
                                             PRIMARY KEY (id, owner));
        """)

    # Files and residency ------------------------------------------------------------------------------
    def _path(self, block_id: str) -> Path:
        digest_prefix = block_id.split("_", 1)[1]
        return self.directory / "blocks" / digest_prefix[:2] / f"{block_id}.safetensors"

    def _write(self, block) -> None:
        path = self._path(block.id)
        path.parent.mkdir(exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.partial")
        if isinstance(block, NativeStateRecord):
            payload = encode_native_state(block.block)
        elif isinstance(block, NativeTensorRecord):
            from ..model.native_artifact import encode_native_tensor_artifact
            payload = encode_native_tensor_artifact(block.artifact)
        else:
            raise TypeError(f"unsupported store record {type(block).__name__}")
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        # The immutable payload must reach durable storage before its SQLite row
        # is published. Persist the rename itself as well as the file contents.
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _size(block) -> int:
        if isinstance(block, NativeStateRecord):
            return sum(tensor.numel() * tensor.element_size() for tensor in block.block.tensors().values())
        if isinstance(block, NativeTensorRecord):
            return sum(tensor.numel() * tensor.element_size() for tensor in block.artifact.tensors.values())
        raise TypeError(f"unsupported store record {type(block).__name__}")

    def _keep(self, block) -> None:
        self._resident[block.id] = block
        self._resident_size += self._size(block)
        if self.directory is None or self.resident_bytes is None:
            return
        while self._resident_size > self.resident_bytes and len(self._resident) > 1:
            _, evicted = self._resident.popitem(last=False)
            self._resident_size -= self._size(evicted)

    def _forget(self, block_id: str) -> None:
        block = self._resident.pop(block_id, None)
        if block is not None:
            self._resident_size -= self._size(block)

    def _known(self, block_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM blocks WHERE id = ?", (block_id,)).fetchone() is not None

    def _load(self, block_id: str):
        found = self._resident.get(block_id)
        if found is not None:
            self._resident.move_to_end(block_id)
            return found
        if self.directory is None or not self._known(block_id):
            return None
        if not (block_id.startswith("nzs1_") or block_id.startswith("nnt1_")):
            raise ValueError(f"unsupported non-native artifact ID in native store: {block_id}")
        try:
            body = self._path(block_id).read_bytes()
            if block_id.startswith("nzs1_"):
                native, actual = decode_native_state(body)
                block = NativeStateRecord(actual, native)
            elif block_id.startswith("nnt1_"):
                from ..model.native_artifact import decode_native_tensor_artifact
                artifact, actual = decode_native_tensor_artifact(body)
                block = NativeTensorRecord(actual, artifact)
            else:  # guarded above
                raise ValueError(f"unsupported non-native artifact ID in native store: {block_id}")
        except FileNotFoundError:
            self._db.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
            return None
        if block.id != block_id:
            raise ValueError(f"{self._path(block_id)} holds {block.id}")
        self._keep(block)
        return block

    def put_native_state(self, block, owner: str | None = None) -> NativeStateRecord:
        """Store raw native state or semantic codes with the same durable lifecycle."""
        from ..model.state_block import NativeStateBlock
        block.validate()
        if block.derivation is not None:
            from ..model.native_state_derivation import verify_state_derivation
            verify_state_derivation(self,block)
        cpu_tensors = {name: tensor.detach().to("cpu").contiguous().clone() for name, tensor in block.tensors().items()}
        stored_block = NativeStateBlock.from_tensors(block.metadata(), cpu_tensors)
        record = NativeStateRecord(native_state_id(stored_block), stored_block)
        return self._put_verified_native_state(record, owner)

    def put_native_states(self, blocks, owner: str | None = None) -> list[NativeStateRecord]:
        """Verify an archived batch before publishing any of its states.

        Every supplied state belongs to the ancestry of a terminal state in this
        batch. Checking those complete ancestries preserves the scalar optimizer
        and moment checks without checking every prefix of a long chain again.
        Inputs are copied first: verification and publication use the same values.
        Existing store records remain available as external dependencies.
        """
        from ..model.state_block import NativeStateBlock
        from ..model.native_state_derivation import verify_state_derivation

        pending = {}
        order = []
        for block in blocks:
            block.validate()
            tensors = {name: tensor.detach().to('cpu').contiguous().clone()
                       for name, tensor in block.tensors().items()}
            # Metadata includes nested derivation bindings; own those as well.
            metadata = json.loads(json.dumps(block.metadata()))
            copied = NativeStateBlock.from_tensors(metadata, tensors)
            ident = native_state_id(copied)
            pending[ident] = NativeStateRecord(ident, copied)
            order.append(ident)

        parents = {ident: record.block.derivation['parent_state_id']
                   for ident, record in pending.items() if record.block.derivation is not None}
        # Prove that terminal coverage is complete, including disconnected cycles.
        done = set()
        publication_order = []
        for root in pending:
            path = set()
            lineage = []
            current = root
            while current in pending and current not in done:
                if current in path:
                    raise ValueError('archived native states have a cyclic derivation')
                path.add(current)
                lineage.append(current)
                current = parents.get(current)
            done.update(path)
            publication_order.extend(reversed(lineage))

        store = self

        class Overlay:
            def get_native_state(self, ident):
                return pending[ident] if ident in pending else store.get_native_state(ident)

            def get_native_tensor_artifact(self, ident):
                return store.get_native_tensor_artifact(ident)

        overlay = Overlay()
        terminals = pending.keys() - set(parents.values())
        for ident in sorted(terminals):
            verify_state_derivation(overlay, pending[ident].block)
        # No state from a failing batch has been admitted above. Publication uses
        # the already verified private copies, with normal collisions and holds.
        published = {ident: self._put_verified_native_state(pending[ident], owner)
                     for ident in publication_order}
        return [published[ident] for ident in order]

    def put_native_archive(self, blocks, artifacts, owner: str | None = None):
        """Verify a complete mixed archive against private copies before admission."""
        from ..model.native_artifact import encode_native_tensor_artifact, native_tensor_artifact_id
        def payloads():
            for block in blocks:
                yield native_state_id(block), encode_native_state(block)
            for artifact in artifacts:
                yield native_tensor_artifact_id(artifact), encode_native_tensor_artifact(artifact)
        return self._put_native_archive_payloads(payloads(), owner)

    def put_native_archive_payloads(self, payloads, owner: str | None = None, *, bounded: bool = False):
        """Decode named immutable archive bodies once, owning every decoded tensor."""
        if bounded:
            return self._put_native_archive_bounded(payloads, owner)
        return self._put_native_archive_payloads(payloads, owner)

    def _put_native_archive_bounded(self, payloads, owner):
        """Stage and verify a complete archive without retaining its tensor inventory."""
        from ..model.native_artifact import decode_native_tensor_artifact, native_tensor_dependencies
        from ..model.state_block import native_state_dependencies
        from ..model.native_derivative_contract import verify_derivative_gradient_origin
        from ..model.native_state_derivation import verify_state_derivation
        if self.directory is None or self.resident_bytes is None:
            raise ValueError('bounded archive import requires a disk-backed bounded store')
        with tempfile.TemporaryDirectory(prefix='archive-', dir=self.directory) as directory:
            stage = TensorStore(directory, self.resident_bytes)
            try:
                ids, kinds, dependencies, parents = [], {}, {}, set()
                for expected, body in payloads:
                    if not isinstance(body, bytes) or not isinstance(expected, str):
                        raise TypeError('native archive payload requires immutable bytes and a content ID')
                    if expected.startswith('nzs1_'):
                        block, ident = decode_native_state(body)
                        record = NativeStateRecord(ident, block)
                        deps = native_state_dependencies(block)
                        if block.derivation is not None:
                            parents.add(block.derivation['parent_state_id'])
                        kind = 'state'
                    elif expected.startswith('nnt1_'):
                        artifact, ident = decode_native_tensor_artifact(body)
                        record = NativeTensorRecord(ident, artifact)
                        deps = native_tensor_dependencies(artifact)
                        kind = 'tensor'
                    else:
                        raise ValueError('archive file does not hold its named content: unsupported native ID')
                    if ident != expected:
                        raise ValueError(f'archive file {expected} does not hold its named content')
                    if ident not in kinds:
                        ids.append(ident)
                    kinds[ident], dependencies[ident] = kind, deps
                    (stage._put_verified_native_state if kind == 'state' else stage._put_verified_native_tensor)(record, None)
                class Overlay:
                    def get_native_state(_, ident):
                        return stage.get_native_state(ident) if ident in kinds else self.get_native_state(ident)
                    def get_native_tensor_artifact(_, ident):
                        return stage.get_native_tensor_artifact(ident) if ident in kinds else self.get_native_tensor_artifact(ident)
                overlay = Overlay()
                done, active, order = set(), set(), []
                for root in ids:
                    stack = [(root, False)]
                    while stack:
                        ident, exiting = stack.pop()
                        if exiting:
                            active.remove(ident)
                            done.add(ident)
                            order.append(ident)
                            continue
                        if ident in done:
                            continue
                        if ident in active:
                            raise ValueError('archived native artifacts have a cyclic dependency')
                        active.add(ident)
                        stack.append((ident, True))
                        for dep in dependencies[ident]:
                            if dep in kinds:
                                stack.append((dep, False))
                            elif dep.startswith('nzs1_'):
                                if overlay.get_native_state(dep) is None:
                                    raise ValueError(f'missing native dependency {dep}')
                            elif dep.startswith('nnt1_'):
                                if overlay.get_native_tensor_artifact(dep) is None:
                                    raise ValueError(f'missing native dependency {dep}')
                            else:
                                raise ValueError(f'invalid native dependency {dep}')
                for ident in ids:
                    if kinds[ident] == 'tensor':
                        verify_derivative_gradient_origin(overlay, overlay.get_native_tensor_artifact(ident).artifact)
                    elif ident not in parents:
                        verify_state_derivation(overlay, overlay.get_native_state(ident).block)
                # Check all collisions before admitting any of the batch.
                for ident in ids:
                    existing = self.get_native_state(ident) if kinds[ident] == 'state' else self.get_native_tensor_artifact(ident)
                    if existing is not None and self._path(ident).read_bytes() != stage._path(ident).read_bytes():
                        raise ValueError('native archive content ID collision')
                for ident in order:
                    record = overlay.get_native_state(ident) if kinds[ident] == 'state' else overlay.get_native_tensor_artifact(ident)
                    # The stage is on the same filesystem. Move the verified immutable
                    # payload into the store instead of serializing a second copy.
                    with self._lock:
                        if not self._known(ident):
                            path = self._path(ident)
                            path.parent.mkdir(exist_ok=True)
                            os.replace(stage._path(ident), path)
                            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                            try:
                                os.fsync(directory_fd)
                            finally:
                                os.close(directory_fd)
                            self._db.execute('INSERT INTO blocks (id, bytes) VALUES (?, ?)', (ident, self._size(record)))
                            self._keep(record)
                        if owner is not None:
                            self._db.execute('INSERT OR IGNORE INTO holds (id, owner) VALUES (?, ?)', (ident, owner))
                return ids
            finally:
                stage.close()

    def _put_native_archive_payloads(self, payloads, owner):
        from ..model.native_artifact import decode_native_tensor_artifact
        states, tensors = {}, {}
        for expected, body in payloads:
            if not isinstance(body, bytes):
                raise TypeError('native archive payload must be immutable bytes')
            if not isinstance(expected, str):
                raise TypeError('native archive payload requires its expected content ID')
            kind = expected[:5]
            if kind == 'nzs1_':
                block, ident = decode_native_state(body)
                states[ident] = NativeStateRecord(ident, block)
            elif kind == 'nnt1_':
                artifact, ident = decode_native_tensor_artifact(body)
                tensors[ident] = NativeTensorRecord(ident, artifact)
            else:
                raise ValueError('archive file does not hold its named content: unsupported native ID')
            if ident != expected:
                raise ValueError(f'archive file {expected} does not hold its named content')
        return self._verify_and_publish_native_archive(states, tensors, owner)

    def _verify_and_publish_native_archive(self, states, tensors, owner):
        from ..model.native_artifact import encode_native_tensor_artifact, native_tensor_dependencies
        from ..model.state_block import native_state_dependencies
        from ..model.native_derivative_contract import verify_derivative_gradient_origin
        from ..model.native_state_derivation import verify_state_derivation
        store = self
        class Overlay:
            def get_native_state(self, ident):
                return states[ident] if ident in states else store.get_native_state(ident)
            def get_native_tensor_artifact(self, ident):
                return tensors[ident] if ident in tensors else store.get_native_tensor_artifact(ident)
        overlay = Overlay()
        records = {**states, **tensors}
        dependencies = {ident: (native_state_dependencies(record.block) if ident in states
                       else native_tensor_dependencies(record.artifact)) for ident, record in records.items()}
        done, active, order = set(), set(), []
        for root in records:
            stack = [(root, False)]
            while stack:
                ident, exiting = stack.pop()
                if exiting:
                    active.remove(ident)
                    done.add(ident)
                    order.append(ident)
                    continue
                if ident in done: continue
                if ident in active: raise ValueError('archived native artifacts have a cyclic dependency')
                active.add(ident)
                stack.append((ident, True))
                for dep in dependencies[ident]:
                    if dep in records: stack.append((dep, False))
                    elif dep.startswith('nzs1_'):
                        if overlay.get_native_state(dep) is None: raise ValueError(f'missing native dependency {dep}')
                    elif dep.startswith('nnt1_'):
                        if overlay.get_native_tensor_artifact(dep) is None: raise ValueError(f'missing native dependency {dep}')
                    else: raise ValueError(f'invalid native dependency {dep}')
        for record in tensors.values(): verify_derivative_gradient_origin(overlay, record.artifact)
        parents = {record.block.derivation['parent_state_id'] for record in states.values()
                   if record.block.derivation is not None}
        for ident in states.keys() - parents: verify_state_derivation(overlay, states[ident].block)
        # Check collisions across the entire batch before publishing any record.
        for ident, record in records.items():
            with store._lock:
                existing = store._load(ident)
            if existing is not None:
                if type(existing) is not type(record):
                    raise ValueError('native archive content ID collision')
                old = encode_native_state(existing.block) if ident in states else encode_native_tensor_artifact(existing.artifact)
                new = encode_native_state(record.block) if ident in states else encode_native_tensor_artifact(record.artifact)
                if old != new: raise ValueError('native archive content ID collision')
        for ident in order:
            if ident in states: self._put_verified_native_state(states[ident], owner)
            else: self._put_verified_native_tensor(tensors[ident], owner)
        return [*states.values(), *tensors.values()]

    def _put_verified_native_state(self, record, owner):
        block = record.block
        with self._lock:
            found = self._load(record.id)
            if found is not None:
                if not isinstance(found, NativeStateRecord) or encode_native_state(found.block) != encode_native_state(block):
                    raise ValueError("native state content ID collision")
                stored = found
            else:
                if self.directory is not None:
                    self._write(record)
                self._db.execute("INSERT INTO blocks (id, bytes) VALUES (?, ?)", (record.id, self._size(record)))
                self._keep(record)
                stored = record
            if owner is not None:
                self._db.execute("INSERT OR IGNORE INTO holds (id, owner) VALUES (?, ?)", (record.id, owner))
            return stored

    def get_native_state(self, block_id: str) -> NativeStateRecord | None:
        with self._lock:
            block = self._load(block_id)
            if block is None:
                return None
            if not isinstance(block, NativeStateRecord):
                raise ValueError(f"{block_id} is a native tensor artifact, not native state")
            return block

    def put_native_tensor_artifact(self, artifact, owner: str | None = None) -> NativeTensorRecord:
        """Store typed named native tensors through the shared durable store and owner lifecycle."""
        from ..model.native_artifact import NativeTensorArtifact, native_tensor_artifact_id
        if not isinstance(artifact, NativeTensorArtifact):
            raise TypeError('expected NativeTensorArtifact')
        artifact.validate()
        from ..model.native_derivative_contract import verify_derivative_gradient_origin
        verify_derivative_gradient_origin(self,artifact)
        cpu_tensors = {name: tensor.detach().to('cpu').contiguous().clone()
                       for name, tensor in artifact.tensors.items()}
        stored_artifact = NativeTensorArtifact(artifact.kind, dict(artifact.native_identity), cpu_tensors,
            artifact.source_artifact_id, dict(artifact.provenance or {}), artifact.canonical_metadata).validate()
        record = NativeTensorRecord(native_tensor_artifact_id(stored_artifact), stored_artifact)
        return self._put_verified_native_tensor(record, owner)

    def _put_verified_native_tensor(self, record, owner):
        stored_artifact = record.artifact
        with self._lock:
            found = self._load(record.id)
            if found is not None:
                from ..model.native_artifact import encode_native_tensor_artifact
                if not isinstance(found, NativeTensorRecord) or encode_native_tensor_artifact(found.artifact) != encode_native_tensor_artifact(stored_artifact):
                    raise ValueError('native tensor artifact content ID collision')
                stored = found
            else:
                if self.directory is not None:
                    self._write(record)
                self._db.execute('INSERT INTO blocks (id, bytes) VALUES (?, ?)', (record.id, self._size(record)))
                self._keep(record)
                stored = record
            if owner is not None:
                self._db.execute('INSERT OR IGNORE INTO holds (id, owner) VALUES (?, ?)', (record.id, owner))
            return stored

    def get_native_tensor_artifact(self, artifact_id: str) -> NativeTensorRecord | None:
        with self._lock:
            artifact = self._load(artifact_id)
            if artifact is None:
                return None
            if not isinstance(artifact, NativeTensorRecord):
                raise ValueError(f'{artifact_id} is not a native tensor artifact')
            return artifact

    # Holds, pins, collection ---------------------------------------------------------------------------
    def hold(self, owner: str, block_ids) -> None:
        """`owner` holds every stored block among `block_ids` (unknown IDs are ignored)."""
        with self._lock:
            self._db.executemany("INSERT OR IGNORE INTO holds (id, owner) SELECT id, ? FROM blocks WHERE id = ?",
                                 [(owner, block_id) for block_id in set(block_ids)])

    def pin(self, block_id: str, owner: str | None = None) -> None:
        """Keep a block through every collection until it is unpinned as often as it was pinned. An owner's pin also
        holds the block, so that owner's collection releases it once unpinned; an anonymous pin is the owner ``""``."""
        with self._lock:
            if not self._known(block_id):
                raise KeyError(block_id)
            self._db.execute("INSERT INTO pins (id, owner, count) VALUES (?, ?, 1) "
                             "ON CONFLICT (id, owner) DO UPDATE SET count = count + 1", (block_id, owner or ""))
            if owner is not None:
                self._db.execute("INSERT OR IGNORE INTO holds (id, owner) VALUES (?, ?)", (block_id, owner))

    def unpin(self, block_id: str, owner: str | None = None) -> None:
        with self._lock:
            self._db.execute("UPDATE pins SET count = count - 1 WHERE id = ? AND owner = ?", (block_id, owner or ""))
            self._db.execute("DELETE FROM pins WHERE count <= 0")

    def _required_closure(self, roots):
        keep = set(roots)
        pending = list(keep)
        for block_id in pending:
            record = self._load(block_id)
            dependencies = []
            if isinstance(record, NativeTensorRecord):
                from ..model.native_artifact import native_tensor_dependencies
                dependencies = native_tensor_dependencies(record.artifact)
            elif isinstance(record, NativeStateRecord):
                from ..model.state_block import native_state_dependencies
                dependencies=native_state_dependencies(record.block)
            for dependency in dependencies:
                if dependency not in keep:
                    keep.add(dependency)
                    pending.append(dependency)
        return keep

    def collect(self, referenced: set[str], owner: str | None = None) -> list[str]:
        """Drop what is no longer needed and return the removed IDs.

        With an `owner`: the owner holds exactly the stored blocks in `referenced` afterwards (and the blocks it pins);
        the blocks it released are dropped unless another owner holds or pins them. Without one: every block that no
        owner holds or pins and `referenced` does not name is dropped."""
        with self._lock:
            db = self._db
            pins = [row[0] for row in db.execute('SELECT id FROM pins')]
            if owner is None:
                roots = [row[0] for row in db.execute('SELECT id FROM holds')] + pins
                referenced = self._required_closure(set(referenced) | set(roots))
                protected = set()
            else:
                own_pins = [row[0] for row in db.execute('SELECT id FROM pins WHERE owner = ?', (owner,))]
                referenced = self._required_closure(set(referenced) | set(own_pins))
                foreign = [row[0] for row in db.execute('SELECT id FROM holds WHERE owner != ?', (owner,))]
                protected = self._required_closure(foreign + pins)
            db.execute("BEGIN")
            try:
                db.execute("CREATE TEMP TABLE IF NOT EXISTS keep (id TEXT PRIMARY KEY)")
                db.execute("DELETE FROM keep")
                db.executemany("INSERT OR IGNORE INTO keep (id) VALUES (?)", [(i,) for i in referenced])
                if owner is None:
                    removed = [row[0] for row in db.execute(
                        "SELECT id FROM blocks WHERE id NOT IN (SELECT id FROM keep) "
                        "AND id NOT IN (SELECT id FROM holds) AND id NOT IN (SELECT id FROM pins) ORDER BY id")]
                else:
                    released = [row[0] for row in db.execute(
                        "SELECT id FROM holds WHERE owner = ? AND id NOT IN (SELECT id FROM keep) "
                        "AND id NOT IN (SELECT id FROM pins WHERE owner = ?) ORDER BY id", (owner, owner))]
                    db.execute("INSERT OR IGNORE INTO holds (id, owner) SELECT id, ? FROM blocks "
                               "WHERE id IN (SELECT id FROM keep)", (owner,))
                    db.executemany("DELETE FROM holds WHERE id = ? AND owner = ?", [(i, owner) for i in released])
                    removed = [i for i in released if db.execute(
                        "SELECT 1 FROM holds WHERE id = ? UNION ALL SELECT 1 FROM pins WHERE id = ?", (i, i)).fetchone() is None]
                removed = [i for i in removed if i not in protected]
                db.executemany("DELETE FROM blocks WHERE id = ?", [(i,) for i in removed])
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            for block_id in removed:
                self._forget(block_id)
                if self.directory is not None:
                    self._path(block_id).unlink(missing_ok=True)
            return removed

    def holders(self, block_id: str) -> set[str]:
        with self._lock:
            return {row[0] for row in self._db.execute("SELECT owner FROM holds WHERE id = ?", (block_id,))}

    @property
    def resident(self) -> int:
        """Native artifacts currently in memory."""
        return len(self._resident)

    def close(self) -> None:
        with self._lock:
            self._db.close()
            if self.directory is not None:
                self._lockfile.close()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
