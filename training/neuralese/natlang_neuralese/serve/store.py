"""Server-side tensor store (S4 §4.1) with the runtime's content addressing.

Block IDs must agree with ts-host's `neuraleseContentId` (ts-host/src/native/neuralese-store.ts):

    nz1_ + base32(sha256("natlang.neuralese-block/1\\0{dialect}\\0{L}x{W}\\0{dtype}\\0" + payload bytes))

with RFC 4648 base32, lower case, no padding. Payloads are stored as little-endian float32 rows on the CPU; the
wire format for one block is a safetensors file with a single `payload` tensor and the block metadata as JSON
under the `natlang.block` metadata key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading
from dataclasses import dataclass, field

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
    meta = json.loads(header.get("__metadata__", {}).get("natlang.block", "{}"))
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
    """Content-addressed, thread-safe, in memory, with pinning and collection."""

    def __init__(self):
        self._blocks: dict[str, Block] = {}
        self._pins: dict[str, int] = {}
        self._lock = threading.Lock()

    def put(self, block: Block) -> Block:
        with self._lock:
            found = self._blocks.get(block.id)
            if found is not None:
                if not found.type and block.type:
                    found.type = block.type
                return found
            self._blocks[block.id] = block
            return block

    def get(self, block_id: str) -> Block | None:
        with self._lock:
            return self._blocks.get(block_id)

    def pin(self, block_id: str) -> None:
        with self._lock:
            if block_id not in self._blocks:
                raise KeyError(block_id)
            self._pins[block_id] = self._pins.get(block_id, 0) + 1

    def unpin(self, block_id: str) -> None:
        with self._lock:
            count = self._pins.get(block_id, 0) - 1
            if count > 0:
                self._pins[block_id] = count
            else:
                self._pins.pop(block_id, None)

    def collect(self, referenced: set[str]) -> list[str]:
        with self._lock:
            removed = [i for i in self._blocks if i not in referenced and i not in self._pins]
            for block_id in removed:
                del self._blocks[block_id]
            return removed

    def __len__(self) -> int:
        return len(self._blocks)
