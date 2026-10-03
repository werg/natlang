"""Deterministic, observational held-out evaluation helpers for training."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch


def canonical_sha256(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def select_fixed_subset(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Choose an input-order-independent deterministic hash sample of held-out IDs."""
    if count < 1:
        raise ValueError("periodic held-out subset size must be positive")
    keyed = []
    seen = set()
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("held-out rows need nonempty stable IDs")
        if row_id in seen:
            raise ValueError(f"duplicate held-out row ID {row_id!r}")
        seen.add(row_id)
        rank = hashlib.sha256(f"periodic-heldout-subset/1\0{seed}\0{row_id}".encode("utf-8")).hexdigest()
        keyed.append((rank, row_id, row))
    return [row for _, _, row in sorted(keyed)[:count]]


def trainable_weights_sha256(model) -> str:
    """Hash only trainable tensors; the frozen base is identified by its model revision."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(parameter.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_fixed_subset(*, model, rows: list[dict], encode: Callable,
                          token_loss_sum: Callable, capture_rng: Callable,
                          restore_rng: Callable, should_stop: Callable[[], bool]):
    """Evaluate rows without changing model mode or RNG, returning token-weighted loss."""
    rng = capture_rng()
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    evaluated = 0
    skipped = 0
    classes: dict[str, int] = {}
    sources: dict[str, int] = {}
    groups: dict[str, int] = {}
    try:
        for row in rows:
            if should_stop():
                break
            encoded = encode(row)
            if encoded is None:
                skipped += 1
                continue
            loss_sum, token_count = token_loss_sum(model, encoded)
            if token_count < 1:
                skipped += 1
                continue
            if not math.isfinite(float(loss_sum)):
                raise ValueError("periodic held-out loss is non-finite")
            total_loss += float(loss_sum)
            total_tokens += int(token_count)
            evaluated += 1
            label = str(row.get("task_family") or row.get("family") or "unknown")
            classes[label] = classes.get(label, 0) + 1
            row_sources = row.get("source_ids") or [row.get("source", "unknown")]
            if not isinstance(row_sources, (list, tuple, set)):
                row_sources = [row_sources]
            for source in set(map(str, row_sources)):
                sources[source] = sources.get(source, 0) + 1
            row_groups = row.get("source_groups") or []
            if not isinstance(row_groups, (list, tuple, set)):
                row_groups = [row_groups]
            for group in set(map(str, row_groups)):
                groups[group] = groups.get(group, 0) + 1
    finally:
        model.train(was_training)
        restore_rng(rng)
    if should_stop():
        return None
    if rows and not evaluated:
        raise ValueError("Every periodic held-out example exceeded --max-len or had no supervised tokens")
    return {
        "loss": total_loss / total_tokens if total_tokens else None,
        "loss_sum": total_loss,
        "supervised_tokens": total_tokens,
        "evaluated_examples": evaluated,
        "skipped_examples": skipped,
        "class_support": dict(sorted(classes.items())),
        "source_support": dict(sorted(sources.items())),
        "source_group_support": dict(sorted(groups.items())),
    }


def read_periodic_metrics(path: Path, *, data_sha256: str, split_sha256: str,
                          model_identity_sha256: str) -> dict[tuple[int, str], dict]:
    """Read and validate append-only metrics, keyed by checkpoint step and subset."""
    completed: dict[tuple[int, str], dict] = {}
    if not path.exists():
        return completed
    raw = path.read_bytes()
    complete_end = raw.rfind(b"\n") + 1
    if complete_end != len(raw):
        tail = raw[complete_end:]
        tail_sha = hashlib.sha256(tail).hexdigest()
        recovery = path.with_name(f"{path.name}.partial-{tail_sha[:16]}.recovered")
        try:
            fd = os.open(recovery, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if recovery.read_bytes() != tail:
                raise ValueError("partial metrics recovery artifact has conflicting bytes")
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(tail)
                stream.flush()
                os.fsync(stream.fileno())
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        fd = os.open(path, os.O_WRONLY)
        try:
            os.ftruncate(fd, complete_end)
            os.fsync(fd)
        finally:
            os.close(fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        raw = raw[:complete_end]
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"periodic held-out metrics line {line_no} is malformed") from exc
        if (not isinstance(item, dict) or
                item.get("schema") != "natlang.periodic_heldout_loss/1" or
                item.get("status") != "completed" or
                item.get("data_sha256") != data_sha256 or
                item.get("split_sha256") != split_sha256 or
                item.get("model_identity_sha256") != model_identity_sha256):
            raise ValueError(f"periodic held-out metrics line {line_no} belongs to another run identity")
        key = (item.get("step"), item.get("subset_ids_sha256"))
        if not isinstance(key[0], int) or not isinstance(key[1], str) or key in completed:
            raise ValueError(f"periodic held-out metrics line {line_no} has invalid or duplicate identity")
        completed[key] = item
    return completed


def periodic_metric_already_complete(completed: dict[tuple[int, str], dict], *,
                                     step: int, subset_ids_sha256: str,
                                     trainable_weights_sha256: str) -> bool:
    """Return whether this exact step/subset was recorded, rejecting weight drift."""
    prior = completed.get((step, subset_ids_sha256))
    if prior is None:
        return False
    if prior.get("trainable_weights_sha256") != trainable_weights_sha256:
        raise ValueError("completed periodic evaluation does not match checkpoint weights at this step")
    return True


def append_periodic_metric(path: Path, item: dict) -> None:
    """Durably append one metric record; reject a duplicate step/subset pair."""
    if (item.get("schema") != "natlang.periodic_heldout_loss/1" or
            item.get("status") != "completed" or
            not isinstance(item.get("step"), int) or item["step"] < 1 or
            not isinstance(item.get("subset_ids_sha256"), str) or
            not isinstance(item.get("trainable_weights_sha256"), str)):
        raise ValueError("periodic held-out metric has an invalid completion identity")
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_periodic_metrics(path, data_sha256=item["data_sha256"],
                                  split_sha256=item["split_sha256"],
                                  model_identity_sha256=item["model_identity_sha256"])
    key = (item["step"], item["subset_ids_sha256"])
    if key in prior:
        raise ValueError("periodic held-out evaluation already exists for this step and subset")
    raw = (json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
