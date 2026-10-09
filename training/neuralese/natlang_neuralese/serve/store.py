"""Server-side tensor store (S4 §4.1) with the runtime's content addressing.

Block IDs must agree with ts-host's `neuraleseContentId` (ts-host/src/native/neuralese-store.ts):

    nz1_ + base32(sha256("natlang.neuralese-block/1\\0{dialect}\\0{L}x{W}\\0{dtype}\\0" + payload bytes))

with RFC 4648 base32, lower case, no padding. Payloads are stored as little-endian float32 rows on the CPU; the
wire format for one block is a safetensors file with a single `payload` tensor and the block metadata as JSON
under the `natlang.block` metadata key.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import sqlite3
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch

HEADER = "natlang.neuralese-block/1"
_DTYPES = {"f32": (torch.float32, "F32", 4), "f16": (torch.float16, "F16", 2), "bf16": (torch.bfloat16, "BF16", 2)}
_FROM_SAFETENSORS = {"F32": "f32", "F16": "f16", "BF16": "bf16"}


def _bytes(tensor: torch.Tensor) -> bytes:
    data = tensor.detach().contiguous().cpu()
    if data.dtype == torch.bfloat16:
        return data.view(torch.int16).numpy().tobytes()
    return data.numpy().tobytes()


def content_id(dialect: str, length: int, width: int, dtype: str, data: bytes) -> str:
    """The runtime's content ID for a block (see the module docstring)."""
    digest = hashlib.sha256(f"{HEADER}\0{dialect}\0{length}x{width}\0{dtype}\0".encode() + data).digest()
    return "nz1_" + base64.b32encode(digest).decode().rstrip("=").lower()


@dataclass
class Block:
    id: str
    dialect: str
    payload: torch.Tensor  # [length, width] float32 on the CPU
    type: str | None = None
    producer: dict = field(default_factory=dict)
    truncated: bool = False
    wire_dtype: str = "f32"
    wire_payload: bytes | None = None  # Preserve the bytes hashed by the uploading client.

    @property
    def length(self) -> int:
        return int(self.payload.shape[0])

    @property
    def width(self) -> int:
        return int(self.payload.shape[1])

    def meta(self) -> dict:
        meta = {"id": self.id, "dialect": self.dialect, "length": self.length, "width": self.width, "dtype": self.wire_dtype}
        if self.type:
            meta["type"] = self.type
        if self.producer:
            meta["producer"] = self.producer
        if self.truncated:
            meta["truncated"] = True
        return meta


def make_block(payload: torch.Tensor, dialect: str, type: str | None = None, producer: dict | None = None,
               truncated: bool = False) -> Block:
    rows = payload.detach().float().cpu().contiguous()
    if rows.dim() != 2:
        raise ValueError(f"a block payload is [length, width], got {tuple(rows.shape)}")
    block_id = content_id(dialect, rows.shape[0], rows.shape[1], "f32", _bytes(rows))
    return Block(block_id, dialect, rows, type, dict(producer or {}), truncated)


def encode_block(block: Block) -> bytes:
    """One block as a safetensors file."""
    data = block.wire_payload if block.wire_payload is not None else _bytes(block.payload)
    header = {
        "__metadata__": {"natlang.block": json.dumps(block.meta(), sort_keys=True)},
        "payload": {"dtype": _DTYPES[block.wire_dtype][1], "shape": [block.length, block.width], "data_offsets": [0, len(data)]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw + data


def decode_block(body: bytes) -> Block:
    """Parse a one-block safetensors file and verify its ID against the content."""
    try:
        return _decode_block(body)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(f"malformed block: {error}") from error


def _decode_block(body: bytes) -> Block:
    if len(body) < 8:
        raise ValueError("not a safetensors body")
    (size,) = struct.unpack("<Q", body[:8])
    if size > len(body) - 8:
        raise ValueError("header exceeds the body")
    header = json.loads(body[8:8 + size])
    if not isinstance(header, dict):
        raise ValueError("block header must be an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("safetensors metadata must be an object")
    meta = json.loads(metadata.get("natlang.block", "{}"))
    if not isinstance(meta, dict):
        raise ValueError("block metadata must be an object")
    entry = header.get("payload")
    if not entry:
        raise ValueError("a block body needs a `payload` tensor")
    dtype = _FROM_SAFETENSORS.get(entry["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported block dtype {entry['dtype']}")
    shape, offsets = entry["shape"], entry["data_offsets"]
    if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int for x in shape):
        raise ValueError("payload shape must contain two integer dimensions")
    length, width = shape
    if length < 0 or width <= 0:
        raise ValueError("block dimensions require nonnegative length and positive width")
    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
        raise ValueError("payload offsets must contain two integers")
    start, end = offsets
    if start < 0 or end < start or end > len(body) - 8 - size:
        raise ValueError("payload offsets exceed the body")
    data = body[8 + size + start:8 + size + end]
    torch_dtype, _, itemsize = _DTYPES[dtype]
    if len(data) != length * width * itemsize:
        raise ValueError("block byte count does not match its shape")
    dialect = meta.get("dialect")
    if not isinstance(dialect, str) or not dialect:
        raise ValueError("block metadata needs a dialect")
    claimed = meta.get("id")
    actual = content_id(dialect, length, width, dtype, data)
    if claimed is not None and claimed != actual:
        raise ValueError(f"block ID {claimed} does not match its content ({actual})")
    if not data:
        tensor = torch.empty(0, dtype=torch_dtype)
    elif dtype == "bf16":
        tensor = torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(data), dtype=torch_dtype)
    rows = tensor.reshape(length, width).float()
    # Inference uses f32; downloads retain the original hash domain and payload bytes.
    return Block(actual, dialect, rows, meta.get("type"), meta.get("producer") or {},
                 bool(meta.get("truncated")), dtype, data)


class TensorStore:
    """Content-addressed, thread-safe block store with owner-scoped holds and pins, optionally spilled to disk.

    Owners are clients (a session or runtime ID, `x-natlang-owner` on the wire). An owner *holds* the blocks it
    uploads, the blocks the server names to it, the blocks it pins and the blocks it lists as referenced; it releases
    them by collecting. A block is dropped only when no owner holds or pins it, so one session's collection never
    removes another's blocks. A collection without an owner drops only blocks that no owner holds or pins.

    With a `directory`, every block is written there content-addressed (`blocks/<xx>/<id>.safetensors`, the wire
    format) when it is first stored, holds and pins live in `store.sqlite` beside them, and only the most recently
    used `resident_bytes` of payloads stay in memory: evicted blocks are reloaded on lookup, and all of it survives a
    restart. One server process owns a directory at a time (an exclusive lock). Without a directory the store is
    purely in memory and nothing is evicted.
    """

    def __init__(self, directory: str | os.PathLike | None = None, resident_bytes: int | None = None):
        self.directory = Path(directory) if directory is not None else None
        self.resident_bytes = resident_bytes
        self._resident: OrderedDict[str, Block] = OrderedDict()
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
        return self.directory / "blocks" / block_id[4:6] / f"{block_id}.safetensors"

    def _write(self, block: Block) -> None:
        path = self._path(block.id)
        path.parent.mkdir(exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.partial")
        partial.write_bytes(encode_block(block))
        os.replace(partial, path)

    @staticmethod
    def _size(block: Block) -> int:
        return block.payload.numel() * block.payload.element_size() + len(block.wire_payload or b"")

    def _keep(self, block: Block) -> None:
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

    def _load(self, block_id: str) -> Block | None:
        found = self._resident.get(block_id)
        if found is not None:
            self._resident.move_to_end(block_id)
            return found
        if self.directory is None or not self._known(block_id):
            return None
        try:
            block = decode_block(self._path(block_id).read_bytes())
        except FileNotFoundError:
            self._db.execute("DELETE FROM blocks WHERE id = ?", (block_id,))
            return None
        if block.id != block_id:
            raise ValueError(f"{self._path(block_id)} holds {block.id}")
        self._keep(block)
        return block

    # Blocks --------------------------------------------------------------------------------------------
    def put(self, block: Block, owner: str | None = None) -> Block:
        """Store `block` (the first block of an ID is kept; a later one may add a missing type); `owner` holds it."""
        with self._lock:
            found = self._load(block.id)
            if found is not None:
                if not found.type and block.type:
                    found.type = block.type
                    if self.directory is not None:
                        self._write(found)
                stored = found
            else:
                if self.directory is not None:
                    self._write(block)
                self._db.execute("INSERT INTO blocks (id, bytes) VALUES (?, ?)", (block.id, self._size(block)))
                self._keep(block)
                stored = block
            if owner is not None:
                self._db.execute("INSERT OR IGNORE INTO holds (id, owner) VALUES (?, ?)", (block.id, owner))
            return stored

    def get(self, block_id: str) -> Block | None:
        with self._lock:
            return self._load(block_id)

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

    def collect(self, referenced: set[str], owner: str | None = None) -> list[str]:
        """Drop what is no longer needed and return the removed IDs.

        With an `owner`: the owner holds exactly the stored blocks in `referenced` afterwards (and the blocks it pins);
        the blocks it released are dropped unless another owner holds or pins them. Without one: every block that no
        owner holds or pins and `referenced` does not name is dropped."""
        with self._lock:
            db = self._db
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
        """Blocks currently in memory."""
        return len(self._resident)

    def close(self) -> None:
        with self._lock:
            self._db.close()
            if self.directory is not None:
                self._lockfile.close()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
