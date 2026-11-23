"""Durable owner-scoped storage for native positional state and named tensors."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import struct
import threading
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The exact tensor inventory comes from NativeStateBlock.tensors().
NATIVE_HEADER = "natlang.native-state/1"
_NATIVE_DTYPES = {torch.float16: ("F16", 2), torch.bfloat16: ("BF16", 2),
                  torch.float32: ("F32", 4), torch.float64: ("F64", 8), torch.int64: ("I64", 8)}
_NATIVE_FROM_SAFE = {value[0]: (key, value[1]) for key, value in _NATIVE_DTYPES.items()}


DEFAULT_RESIDENT_BYTES = 2 << 30  # resident payload bytes of a directory store (serve --store-resident-gb)


def _tensor_view(tensor: torch.Tensor) -> np.ndarray:
    """The tensor's bytes as a flat uint8 array: no copy for a contiguous CPU tensor."""
    data = tensor.detach()
    if data.device.type != "cpu":
        data = data.cpu()
    data = data.contiguous()
    if data.dtype == torch.bfloat16:
        data = data.view(torch.int16)
    return data.reshape(-1).numpy().view(np.uint8)


def _native_layout(tensors) -> list[tuple[str, str, list[int]]]:
    """[(name, safetensors dtype, shape)] in the canonical (name) order of ids and files."""
    layout = []
    for name, tensor in sorted(tensors.items()):
        dtype = _NATIVE_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"unsupported native state dtype {tensor.dtype}")
        layout.append((name, dtype[0], list(tensor.shape)))
    return layout


def _native_digest(metadata: dict, layout, chunks) -> str:
    """The content id over canonical metadata, ordered descriptors and every tensor byte; `chunks(name)` yields the
    named tensor's bytes in order (whole in memory, or streamed from a file)."""
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    digest.update(NATIVE_HEADER.encode() + b"\0" + struct.pack("<Q", len(encoded)) + encoded)
    for name, dtype, shape in layout:
        encoded_name = name.encode()
        descriptor = json.dumps({"dtype": dtype, "shape": shape}, separators=(",", ":")).encode()
        digest.update(struct.pack("<I", len(encoded_name)) + encoded_name)
        digest.update(struct.pack("<I", len(descriptor)) + descriptor)
        digest.update(struct.pack("<Q", math.prod(shape) * _NATIVE_FROM_SAFE[dtype][1]))
        for chunk in chunks(name):
            digest.update(chunk)
    return "nzs1_" + base64.b32encode(digest.digest()).decode().rstrip("=").lower()


def _native_header(block_id: str, metadata: dict, layout) -> bytes:
    """The safetensors prefix (length + JSON header, padded to 8 bytes) of a native-state file."""
    entries, offset = {}, 0
    for name, dtype, shape in layout:
        size = math.prod(shape) * _NATIVE_FROM_SAFE[dtype][1]
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    described = json.dumps({"id": block_id, **metadata}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    entries["__metadata__"] = {"natlang.native_state": described}
    header = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header


def native_state_id(block) -> str:
    """Cross-language ID over canonical provenance, ordered descriptors and every tensor byte."""
    tensors = block.tensors()
    views = {name: _tensor_view(tensor) for name, tensor in tensors.items()}
    return _native_digest(block.metadata(), _native_layout(tensors), lambda name: (views[name],))


@dataclass
class NativeTensorRecord:
    """Named native parameter/gradient/update artifact in the shared store."""
    id: str
    artifact: object

    def meta(self):
        return {'id': self.id, **self.artifact.metadata()}


def encode_native_state(block) -> bytes:
    """Encode a validated NativeStateBlock as deterministic multi-tensor safetensors (the bytes of its store file)."""
    from ..model.state_block import NativeStateBlock
    if not isinstance(block, NativeStateBlock):
        raise TypeError("expected NativeStateBlock")
    block.validate()
    tensors = block.tensors()
    views = {name: _tensor_view(tensor) for name, tensor in tensors.items()}
    layout = _native_layout(tensors)
    metadata = block.metadata()
    block_id = _native_digest(metadata, layout, lambda name: (views[name],))
    return b"".join([_native_header(block_id, metadata, layout), *(views[name] for name, _, _ in layout)])


def _fsync(path) -> None:
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if os.path.isdir(path) else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_native_state_file(path, metadata: dict, tensors: dict, *, durable: bool = False) -> str:
    """Write native state as its store file at `path` (a partial file renamed into place) and return its id.

    The one writer of the format `encode_native_state` encodes and `read_native_state_file` reads: the store writes
    blocks with it, and the SGLang fork writes a produce span's capture with it (`custom_params.natlang_capture`
    `native_state`: the front's metadata), so the front adopts the fork's file into the store without decoding it.
    `metadata` is `NativeStateBlock.metadata()`; nothing is validated here (the reader checks everything). `durable`
    fsyncs the payload before the rename and the directory after it."""
    views = {name: _tensor_view(tensor) for name, tensor in tensors.items()}
    layout = _native_layout(tensors)
    block_id = _native_digest(metadata, layout, lambda name: (views[name],))
    path = os.fspath(path)
    partial = f"{path}.{os.getpid()}.{threading.get_ident()}.partial"
    try:
        with open(partial, "wb") as stream:
            stream.write(_native_header(block_id, metadata, layout))
            for name, _, _ in layout:
                stream.write(views[name])
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    if durable:
        _fsync(os.path.dirname(path) or ".")
    return block_id


def _parse_native(buffer):
    """(metadata with id, [(name, dtype, shape, begin, end)] sorted by offset with absolute byte offsets) of a native
    state's bytes (bytes or a file mapping), reading the header only; rejects gaps, overlaps and trailing bytes."""
    if len(buffer) < 8:
        raise ValueError("not a safetensors body")
    (header_size,) = struct.unpack("<Q", buffer[:8])
    if header_size > len(buffer) - 8:
        raise ValueError("header exceeds the body")
    header = json.loads(bytes(buffer[8:8 + header_size]))
    if not isinstance(header, dict):
        raise ValueError("native state header must be an object")
    metadata = json.loads(header.get("__metadata__", {}).get("natlang.native_state", "{}"))
    if not isinstance(metadata, dict):
        raise ValueError("native state metadata must be an object")
    base, payload = 8 + header_size, len(buffer) - 8 - header_size
    entries = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype_info = _NATIVE_FROM_SAFE.get(entry["dtype"])
        if dtype_info is None:
            raise ValueError(f"unsupported native state dtype {entry['dtype']}")
        shape, offsets = entry["shape"], entry["data_offsets"]
        if (not isinstance(shape, list) or not shape or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets)):
            raise ValueError(f"invalid native tensor descriptor: {name}")
        start, end = offsets
        if start < 0 or end < start or end > payload or end - start != math.prod(shape) * dtype_info[1]:
            raise ValueError(f"invalid native tensor byte range: {name}")
        entries.append((name, entry["dtype"], shape, base + start, base + end))
    entries.sort(key=lambda entry: (entry[3], entry[4]))
    if (not entries or entries[0][3] != base or entries[-1][4] != len(buffer)
            or any(a[4] != b[3] for a, b in zip(entries, entries[1:]))):
        raise ValueError("native state tensor payload has gaps or overlaps")
    return metadata, entries


def _array(buffer, dtype: str, shape) -> torch.Tensor:
    """One owned tensor copied from a byte range."""
    torch_dtype = _NATIVE_FROM_SAFE[dtype][0]
    if not len(buffer):   # an empty span (a zero-length byte array cannot be viewed as wider elements)
        return torch.empty(shape, dtype=torch_dtype)
    raw = np.frombuffer(buffer, dtype=np.uint8).copy()
    if torch_dtype == torch.bfloat16:
        return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(raw).view(torch_dtype).reshape(shape)


def decode_native_state(body: bytes):
    """Decode and verify a native state artifact; reject gaps, overlaps, unknown tensors and false IDs."""
    from ..model.state_block import NativeStateBlock
    try:
        metadata, entries = _parse_native(body)
        view = memoryview(body)
        tensors = {name: _array(view[begin:end], dtype, shape) for name, dtype, shape, begin, end in entries}
        block = NativeStateBlock.from_tensors(metadata, tensors)
        actual = native_state_id(block)
        if metadata.get("id") != actual:
            raise ValueError(f"native state ID does not match its content ({actual})")
        return block, actual
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed native state: {error}") from error


# Exponent masks: a float is non-finite iff all its exponent bits are set.
_NONFINITE = {"BF16": (np.uint16, 0x7F80), "F16": (np.uint16, 0x7C00), "F32": (np.uint32, 0x7F800000),
              "F64": (np.uint64, 0x7FF0000000000000)}
_STREAM_BYTES = 32 << 20


class NativeStateFile:
    """A native state read from its file's header and its two small index tensors (token_ids, positions), never its
    payload: identity, layout and text for whoever moves the state without computing on it (the SGLang front hands the
    path to the fork, the HTTP server streams the file). Both a store record (`id`, `meta()`) and its block (`block`
    is itself; the identity fields, `text`, `token_ids`, `adapter_binding`, `derivation`, `producer_kind` as on
    `NativeStateBlock`).
    `load()` makes the `NativeStateBlock` for code that computes on the tensors."""

    def __init__(self, path, metadata, entries, token_ids, positions, stat):
        self.path = Path(path)
        self.id = metadata["id"]
        self._metadata = {key: value for key, value in metadata.items() if key != "id"}
        self.entries = entries
        self.token_ids, self.positions = token_ids, positions
        self.file_key = (stat.st_dev, stat.st_ino, stat.st_size)
        self.nbytes = sum(end - begin for _, _, _, begin, end in entries)

    text = property(lambda self: self._metadata["text"])
    architecture = property(lambda self: self._metadata["architecture"])
    tokenizer_fingerprint = property(lambda self: self._metadata["tokenizer_fingerprint"])
    checkpoint_sha256 = property(lambda self: self._metadata["checkpoint_sha256"])
    embedding_weight_sha256 = property(lambda self: self._metadata["embedding_weight_sha256"])
    adapter_binding = property(lambda self: self._metadata.get("adapter_binding"))
    derivation = property(lambda self: self._metadata.get("derivation"))
    producer_kind = property(lambda self: self._metadata.get("producer_kind"))
    layer_kinds = property(lambda self: tuple(self._metadata["layer_kinds"]))
    block = property(lambda self: self)

    def dtype(self, name):
        return next(_NATIVE_FROM_SAFE[dtype][0] for entry, dtype, _, _, _ in self.entries if entry == name)

    def metadata(self) -> dict:
        return json.loads(json.dumps(self._metadata))

    def meta(self) -> dict:
        return {"id": self.id, **self.metadata()}

    def validate(self):
        return self   # checked when read (`read_native_state_file`)

    def require_native(self, *, checkpoint_sha256, tokenizer_fingerprint, embedding_weight_sha256,
                       architecture, layer_kinds=None) -> None:
        """`NativeStateBlock.require_native` from the header."""
        for name, expected in (('checkpoint_sha256', checkpoint_sha256),
                               ('tokenizer_fingerprint', tokenizer_fingerprint),
                               ('embedding_weight_sha256', embedding_weight_sha256),
                               ('architecture', architecture)):
            if getattr(self, name) != expected:
                raise ValueError(f'foreign native state {name}: adapt or reencode its text explicitly')
        if layer_kinds is not None and self.layer_kinds != tuple(layer_kinds):
            raise ValueError('native state layer layout differs from the consuming backbone')

    def load(self):
        """The `NativeStateBlock`, each tensor copied once from the file mapping. The file must still be the one that
        was verified (store files are only ever renamed into place, never rewritten)."""
        import mmap
        from ..model.state_block import NativeStateBlock
        with open(self.path, "rb") as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino, stat.st_size) != self.file_key:
                raise ValueError(f"{self.path} changed since it was verified")
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                view = memoryview(mapped)
                try:
                    tensors = {name: _array(view[begin:end], dtype, shape)
                               for name, dtype, shape, begin, end in self.entries}
                finally:
                    view.release()
        return NativeStateBlock.from_tensors(self._metadata, tensors)


class NativeStateIdMismatch(ValueError):
    """A native-state file holds another state than the one it was named for."""


def read_native_state_file(path, expected_id: str | None = None) -> NativeStateFile:
    """Verify a native-state file without materializing its payload: the header's metadata and layout
    (`validate_native_layout`), the index tensors' values, the adapter binding, and one streaming pass over the payload
    that hashes it into the content id and checks every float is finite. Raises ValueError (FileNotFoundError when the
    file is missing)."""
    import mmap
    from ..model.state_block import validate_native_layout, check_token_values
    with open(path, "rb") as stream:
        stat = os.fstat(stream.fileno())
        if stat.st_size < 8:
            raise ValueError(f"{path}: not a safetensors body")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            view = memoryview(mapped)
            try:
                try:
                    metadata, entries = _parse_native(view)
                except (KeyError, TypeError, IndexError, OverflowError, AttributeError, json.JSONDecodeError) as error:
                    raise ValueError(f"malformed native state: {error}") from error
                plain = {key: value for key, value in metadata.items() if key != "id"}
                by_name = {entry[0]: entry for entry in entries}
                validate_native_layout(plain, {name: (_NATIVE_FROM_SAFE[dtype][0], tuple(shape))
                                               for name, dtype, shape, _, _ in entries})
                index = {name: _array(view[by_name[name][3]:by_name[name][4]], "I64", by_name[name][2])
                         for name in ("token_ids", "positions")}
                check_token_values(index["token_ids"], index["positions"])

                def chunks(name):
                    _, dtype, _, begin, end = by_name[name]
                    for offset in range(begin, end, _STREAM_BYTES):
                        chunk = view[offset:min(offset + _STREAM_BYTES, end)]
                        try:
                            if dtype in _NONFINITE:
                                kind, mask = _NONFINITE[dtype]
                                bits = np.frombuffer(chunk, dtype=kind)
                                finite = not bool(((bits & kind(mask)) == kind(mask)).any())
                                del bits
                                if not finite:
                                    raise ValueError(f"native state {name} contains non-finite values")
                            yield chunk
                        finally:
                            chunk.release()

                layout = [(name, dtype, shape) for name, dtype, shape, _, _ in sorted(entries)]
                actual = _native_digest(plain, layout, chunks)
            finally:
                view.release()
    if metadata.get("id") != actual:
        raise ValueError(f"{path} holds native state {actual}, not {metadata.get('id')}")
    if expected_id is not None and actual != expected_id:
        raise NativeStateIdMismatch(f"{path} holds native state {actual}, not {expected_id}")
    state = NativeStateFile(path, metadata, entries, index["token_ids"], index["positions"], stat)
    if state.adapter_binding is not None:
        from ..model.native_adapters import validate_state_binding
        validate_state_binding(state.adapter_binding, state)
    return state


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
        # A directory store keeps its payloads on disk, so residency is a bounded LRU by default: unbounded, the GRPO
        # actor front (serve_native_checkpoint.py) held every block it ever stored, ~8 GB of RSS by grpo-v1-run2 step 5.
        self.resident_bytes = (DEFAULT_RESIDENT_BYTES if resident_bytes is None and self.directory is not None
                               else resident_bytes)
        self._resident: OrderedDict[str, NativeStateRecord | NativeTensorRecord] = OrderedDict()
        self._resident_size = 0
        # Verified native-state files (header and index tensors only), by id: a file is verified once per process
        # (`read_native_state_file`); store files are immutable, only ever renamed into place.
        self._files: OrderedDict[str, NativeStateFile] = OrderedDict()
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
        if isinstance(block, NativeStateRecord):
            # The immutable payload reaches durable storage before its SQLite row is published.
            write_native_state_file(path, block.block.metadata(), block.block.tensors(), durable=True)
            return
        partial = path.with_suffix(f".{os.getpid()}.partial")
        if isinstance(block, NativeTensorRecord):
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
        self._files.pop(block_id, None)
        block = self._resident.pop(block_id, None)
        if block is not None:
            self._resident_size -= self._size(block)

    def _known(self, block_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM blocks WHERE id = ?", (block_id,)).fetchone() is not None

    def _touch(self, record) -> None:
        """Mark a stored block as used now (its file's mtime): a trainer that reads this store's files collects those
        unused since its oldest live episode (scripts/neuralese_data/grpo.py collect_blocks). A file deleted that way
        while the block stays resident is written again."""
        if self.directory is None:
            return
        try:
            os.utime(self._path(record.id))
        except FileNotFoundError:
            self._write(record)

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
            if block_id.startswith("nzs1_"):
                state = self._file_locked(block_id)
                if state is None:
                    return None
                block = NativeStateRecord(state.id, state.load())
            elif block_id.startswith("nnt1_"):
                body = self._path(block_id).read_bytes()
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

    # Native-state files: identity and layout without the payload ---------------------------------------------
    _FILES_KEPT = 4096

    def _remember(self, state: NativeStateFile) -> None:
        self._files[state.id] = state
        self._files.move_to_end(state.id)
        while len(self._files) > self._FILES_KEPT:
            self._files.popitem(last=False)

    def _cached_file(self, block_id: str):
        """The verified view of a known block's file if that file is unchanged, else None (and a vanished file's row is
        dropped, rewritten first when the block is resident). Holds the lock."""
        path = self._path(block_id)
        try:
            stat = path.stat()
        except FileNotFoundError:
            resident = self._resident.get(block_id)
            if resident is None:
                self._files.pop(block_id, None)
                self._db.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
                raise
            self._write(resident)
            stat = path.stat()
        state = self._files.get(block_id)
        if state is not None and state.file_key == (stat.st_dev, stat.st_ino, stat.st_size):
            self._files.move_to_end(block_id)
            return state
        return None

    def _file_locked(self, block_id: str):
        try:
            state = self._cached_file(block_id)
        except FileNotFoundError:
            return None
        if state is None:
            try:
                state = read_native_state_file(self._path(block_id), block_id)
            except FileNotFoundError:
                self._db.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
                return None
            self._remember(state)
        return state

    def native_state_file(self, block_id: str) -> NativeStateFile | None:
        """A stored native state as its verified file (`NativeStateFile`: metadata, layout, token ids and path, no
        payload), marked used now; None when unknown or for a store without a directory (whose states live only in
        memory: `get_native_state`). A file is verified once per process, outside the store lock."""
        if self.directory is None:
            return None
        if not block_id.startswith("nzs1_"):
            raise ValueError(f"{block_id} is not native state")
        with self._lock:
            if not self._known(block_id):
                return None
            try:
                state = self._cached_file(block_id)
            except FileNotFoundError:
                return None
        if state is None:
            try:
                state = read_native_state_file(self._path(block_id), block_id)
            except FileNotFoundError:
                with self._lock:
                    self._files.pop(block_id, None)
                    self._db.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
                return None
            with self._lock:
                self._remember(state)
        try:
            os.utime(state.path)   # recency for collect_blocks, as `_touch`
        except FileNotFoundError:
            return self.native_state_file(block_id)
        return state

    def adopt_native_state_file(self, path, owner: str | None = None, *, expected_id: str | None = None,
                                metadata: dict | None = None, token_ids=None) -> NativeStateFile:
        """Admit a native-state file written in the store's format (`write_native_state_file`, e.g. by the SGLang fork)
        by renaming it into the store, verified from the file (`read_native_state_file`: layout, values, content id)
        without decoding its payload. `expected_id`: the id it was named for (`NativeStateIdMismatch`); `metadata`,
        `token_ids`: what the writer was asked to store (refused when the file says otherwise). The file must be on
        the store's filesystem; it is consumed (moved or, when the state is already stored with identical bytes,
        deleted) unless refused."""
        import filecmp
        if self.directory is None:
            raise ValueError("adopting a native-state file needs a directory store")
        path = Path(path)
        state = read_native_state_file(path, expected_id)
        if metadata is not None and state.metadata() != json.loads(json.dumps(metadata)):
            raise ValueError("native state file metadata differs from the requested state")
        if token_ids is not None and state.token_ids.tolist() != [int(token) for token in token_ids]:
            raise ValueError("captured producer token IDs differ from the declared source span")
        if state.derivation is not None:
            from ..model.native_state_derivation import verify_state_derivation
            verify_state_derivation(self, state.load())
        target = self._path(state.id)
        with self._lock:
            known, existing = self._known(state.id), None
            if known:
                try:
                    existing = self._cached_file(state.id)
                except FileNotFoundError:   # the row went with its file
                    known = False
            if known:
                if not filecmp.cmp(path, target, shallow=False):
                    raise ValueError("native state content ID collision")
                path.unlink()
                os.utime(target)
                stored = existing or read_native_state_file(target, state.id)
            else:
                target.parent.mkdir(exist_ok=True)
                _fsync(path)
                os.replace(path, target)   # same filesystem: the capture directory lies inside the store
                _fsync(target.parent)
                self._db.execute("INSERT INTO blocks (id, bytes) VALUES (?, ?)", (state.id, state.nbytes))
                stat = target.stat()
                stored = NativeStateFile(target, state.meta(), state.entries, state.token_ids, state.positions, stat)
            self._remember(stored)
            if owner is not None:
                self._db.execute("INSERT OR IGNORE INTO holds (id, owner) VALUES (?, ?)", (state.id, owner))
        return stored

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
                self._touch(found)
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
            self._touch(block)
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

    def contains(self, block_id: str) -> bool:
        """Check the stored inventory without loading tensor payloads."""
        with self._lock:
            return self._known(block_id)

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
