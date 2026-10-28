"""Fragment planning and outer-optimizer numerics shared by DiLoCo learners and coordinator.

Ported from megatoken's Decoupled DiLoCo (``megatoken.training.diloco_protocol``)
into a model-agnostic form: a trainer exposes its trainable FP32 parameters
through a :class:`ParameterSurface` and never needs to share an optimizer class.
Only trainable floating-point parameters participate in outer optimization;
inner optimizer moments (AdamW, MuonSR, ...) stay local to each learner. Wire
data uses safetensors, never pickle.
"""

from __future__ import annotations

import math
import json
import os
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from safetensors.torch import load_file

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_hex

PROTOCOL = 1


@dataclass(frozen=True)
class Slice:
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class Fragment:
    index: int
    slices: tuple[Slice, ...]

    @property
    def length(self) -> int:
        return sum(part.length for part in self.slices)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "slices": [asdict(item) for item in self.slices]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Fragment:
        return cls(index=int(value["index"]), slices=tuple(Slice(**item) for item in value["slices"]))


class ParameterSurface(Protocol):
    """What a trainer exposes to DiLoCo.

    ``masters`` are the authoritative trainable tensors the inner optimizer
    updates (FP32 masters where the trainer keeps them, otherwise the
    parameters themselves). ``after_install`` propagates freshly written
    master values to any separately served copies (for example BF16 model
    weights behind FP32 masters). ``frozen_names`` lists masters that this
    stage does not train; a fragment made only of them is never sent.
    """

    masters: Mapping[str, torch.Tensor]
    frozen_names: frozenset[str]

    def after_install(self, names: set[str]) -> None: ...


@dataclass
class TensorSurface:
    """Surface over plain trainable tensors that are their own masters (for example FP32 LFM)."""

    masters: Mapping[str, torch.Tensor]
    frozen_names: frozenset[str] = frozenset()

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> TensorSurface:
        return cls(masters={n: p for n, p in module.named_parameters() if p.requires_grad and torch.is_floating_point(p)})

    def after_install(self, names: set[str]) -> None:
        pass


def parameter_identity(masters: Mapping[str, torch.Tensor]) -> str:
    return canonical_json_sha256_hex({name: list(p.shape) for name, p in sorted(masters.items())})


def data_shard(row_id: str, learners: int) -> int:
    """The learner that trains one source row: a stable hash, independent of row order and appends."""
    return int(sha256_hex(str(row_id).encode("utf-8"))[:16], 16) % learners


def plan_fragments(masters: Mapping[str, torch.Tensor], *, target_bytes: int = 64 << 20) -> tuple[Fragment, ...]:
    """Deterministic, balanced byte-sized fragments; large tensors split at flat offsets."""
    if target_bytes < 1024 or target_bytes % 4:
        raise ValueError("fragment size must be >=1024 bytes and a multiple of four")
    capacity = target_bytes // 4
    result: list[Fragment] = []
    current: list[Slice] = []
    remaining = capacity

    def emit() -> None:
        nonlocal current, remaining
        if current:
            result.append(Fragment(len(result), tuple(current)))
            current = []
            remaining = capacity

    for name, param in sorted(masters.items()):
        if not torch.is_floating_point(param):
            raise ValueError(f"Nonfloating master parameter {name}")
        start = 0
        while start < param.numel():
            count = min(remaining, param.numel() - start)
            current.append(Slice(name, start, count))
            start += count
            remaining -= count
            if not remaining:
                emit()
    emit()
    if not result:
        raise ValueError("A DiLoCo run needs trainable parameters")
    return tuple(result)


@torch.no_grad()
def pack_fragment(masters: Mapping[str, torch.Tensor], fragment: Fragment) -> torch.Tensor:
    """Snapshot one fragment as FP32 host memory; call only at an optimizer-step boundary.

    Device slices land in one pinned buffer through non-blocking copies and a
    single synchronization (pageable per-slice copies are >20x slower).
    """
    on_device = any(masters[part.name].device.type == "cuda" for part in fragment.slices)
    packed = torch.empty(fragment.length, dtype=torch.float32, pin_memory=on_device)
    cursor = 0
    for part in fragment.slices:
        source = masters[part.name].detach().reshape(-1)[part.offset : part.offset + part.length]
        packed[cursor : cursor + part.length].copy_(source, non_blocking=on_device)
        cursor += part.length
    if on_device:
        torch.cuda.synchronize()
    return packed


@torch.no_grad()
def install_fragment(surface: ParameterSurface, fragment: Fragment, values: torch.Tensor) -> None:
    """Overwrite masters (and served copies); inner optimizer moments are preserved."""
    if values.ndim != 1 or values.numel() != fragment.length or not values.isfinite().all():
        raise ValueError("Invalid synchronized parameter fragment")
    cursor = 0
    touched: set[str] = set()
    for part in fragment.slices:
        master = surface.masters[part.name]
        master.detach().reshape(-1)[part.offset : part.offset + part.length].copy_(
            values[cursor : cursor + part.length].to(master.device, master.dtype)
        )
        cursor += part.length
        touched.add(part.name)
    surface.after_install(touched)


def merge_pseudogradients(
    deltas: list[torch.Tensor], tokens: list[int], steps: list[int], staleness: list[int]
) -> torch.Tensor:
    """Token quantity x tokens-per-step quality, with stale down-weighting (Decoupled DiLoCo).

    A pseudogradient is baseline minus learner weights. Normalization is over
    the contributors, not the configured learner count: failed learners have no vote.
    """
    if not deltas or not (len(deltas) == len(tokens) == len(steps) == len(staleness)):
        raise ValueError("Malformed outer gradient quorum")
    if any(t <= 0 or s <= 0 or age < 0 for t, s, age in zip(tokens, steps, staleness, strict=True)):
        raise ValueError("Outer gradients require positive counts and nonnegative age")
    if any(d.shape != deltas[0].shape or not torch.isfinite(d).all() for d in deltas):
        raise ValueError("Invalid outer gradient tensor")
    weights = [
        (float(t) * (float(t) / float(s))) / (1.0 + float(age))
        for t, s, age in zip(tokens, steps, staleness, strict=True)
    ]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Invalid merge weight")
    total = sum(weights)
    return sum(
        (d.float() * (w / total) for d, w in zip(deltas, weights, strict=True)),
        torch.zeros_like(deltas[0], dtype=torch.float32),
    )


def ema_decay_at(decay: float, warmup: int, updates: int) -> float:
    """EMA decay used by the update that follows ``updates`` earlier ones (``warmup`` N >= 2: min(decay, (1+n)/(N+n)))."""
    if warmup >= 2:
        return min(decay, (1 + updates) / (warmup + updates))
    return decay


def outer_nesterov(
    theta: torch.Tensor, velocity: torch.Tensor, pseudo_gradient: torch.Tensor, *, lr: float, momentum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (lr > 0 and 0 <= momentum < 1):
        raise ValueError("Invalid outer Nesterov hyperparameters")
    velocity = velocity * momentum + pseudo_gradient
    next_theta = theta - lr * (pseudo_gradient + momentum * velocity)
    if not torch.isfinite(next_theta).all():
        raise ValueError("Outer update generated nonfinite parameters")
    return next_theta, velocity


def save_tensors(
    path: str | Path, tensors: Mapping[str, torch.Tensor], metadata: dict[str, str] | None = None,
    *, dtype: torch.dtype | None = None,
) -> None:
    """Atomically publish safetensors with bounded conversion and dirty-page buffers.

    Seeds span the whole model, unlike wire fragments. Never serialize their
    payload into a single Python bytes object or convert every tensor at once.
    ``dtype`` casts each output chunk, leaving mmap-backed source tensors intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = {torch.bool: 'BOOL', torch.uint8: 'U8', torch.int8: 'I8',
             torch.int16: 'I16', torch.int32: 'I32', torch.int64: 'I64',
             torch.float16: 'F16', torch.bfloat16: 'BF16',
             torch.float32: 'F32', torch.float64: 'F64'}
    header: dict[str, Any] = {}
    if metadata is not None:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise ValueError('safetensors metadata must contain strings')
        header['__metadata__'] = metadata
    offset = 0
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or name == '__metadata__':
            raise ValueError('invalid safetensors tensor name')
        target_dtype = dtype or tensor.dtype
        if target_dtype not in codes or tensor.layout != torch.strided:
            raise ValueError('unsupported safetensors tensor dtype or layout')
        size = tensor.numel() * torch.empty((), dtype=target_dtype).element_size()
        header[name] = dict(dtype=codes[target_dtype], shape=list(tensor.shape), data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    window = 16 << 20
    fd, pending = tempfile.mkstemp(prefix=path.name + '.', suffix='.pending', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb', buffering=0) as stream:
            released = 0

            def write(data):
                nonlocal released
                view = memoryview(data).cast('B')
                while len(view):
                    remaining = window - (stream.tell() - released)
                    count = stream.write(view[:remaining])
                    if not count:
                        raise OSError('safetensors write made no progress')
                    view = view[count:]
                    if stream.tell() - released >= window:
                        os.fdatasync(stream.fileno())
                        if hasattr(os, 'posix_fadvise'):
                            os.posix_fadvise(stream.fileno(), released, stream.tell() - released, os.POSIX_FADV_DONTNEED)
                        released = stream.tell()

            write(struct.pack('<Q', len(encoded)))
            write(encoded)
            for name, tensor in sorted(tensors.items()):
                # Noncontiguous callers need one contiguous tensor, never a
                # model-wide copy. Native checkpoint parameters are contiguous.
                flat = tensor.detach().reshape(-1)
                target_dtype = dtype or tensor.dtype
                elements = max(1, window // max(tensor.element_size(), torch.empty((), dtype=target_dtype).element_size()))
                for start in range(0, flat.numel(), elements):
                    chunk = flat[start:start + elements].to(device='cpu', dtype=target_dtype).contiguous()
                    write(chunk.view(torch.uint8).numpy())
            os.fsync(stream.fileno())
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), released, 0, os.POSIX_FADV_DONTNEED)
        os.replace(pending, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(pending).unlink(missing_ok=True)


def load_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")
