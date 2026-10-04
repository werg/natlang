"""A bounded, deterministic, resumable stream of rendered port records for full-corpus runs (S3 full-run handoff §4).

The pilot's `load_family` materialises a family; this stream holds only a byte-offset index:

- Index: one pass per family file records each line's byte offset, split, outcome label and source group (hashed).
  It is cached beside an identity of the file (size, mtime) and refused if the file changes: a running dataset is
  never silently replaced. Later data enters through a new stream at a recorded boundary.
- Source-group sampling: an epoch visits a seeded permutation of the family's groups and takes one record per
  group, cycling through a group's records across epochs, so large groups of near-duplicates do not dominate.
- Families interleave by smooth weighted round-robin, which needs no random state.
- Length bounds: a record whose producer, target or consumer view exceeds its bound is held (counted and logged
  by ID), never truncated into a positive.
- Resume: `state_dict()` is a few integers per family; `load_state_dict` restores the exact next record.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .records import RecordError, parse_record
from .render import RenderedRecord, Renderer, render_record

INDEX_VERSION = "natlang.port-record-index/1"


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little", signed=True)


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_index(path: str | Path, index_dir: str | Path) -> Path:
    """Index one port-record JSONL file (offsets, split, label, group); reuses a matching cached index."""
    path, index_dir = Path(path), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    target = index_dir / (path.name + ".index.npz")
    identity = _identity(path)
    if target.exists():
        with np.load(target, allow_pickle=False) as cached:
            if json.loads(str(cached["meta"])).get("identity") == identity:
                return target
    offsets, splits, labels, groups = [], [], [], []
    label_names: dict[str, int] = {}
    split_names: dict[str, int] = {}
    invalid = 0
    with open(path, "rb") as handle:
        offset = 0
        for line in handle:
            start, offset = offset, offset + len(line)
            if not line.strip():
                continue
            try:
                record = parse_record(json.loads(line))
            except (RecordError, json.JSONDecodeError):
                invalid += 1
                continue
            offsets.append(start)
            splits.append(split_names.setdefault(record.split, len(split_names)))
            labels.append(label_names.setdefault(record.outcome_label, len(label_names)))
            groups.append(_hash64(record.split_groups[0] if record.split_groups else "id:" + record.id))
    if _identity(path) != identity:
        raise ValueError(f"{path} changed while it was being indexed")
    meta = {"version": INDEX_VERSION, "identity": identity, "labels": label_names, "splits": split_names,
            "invalid_rows": invalid}
    pending = target.with_suffix(".pending.npz")
    np.savez(pending, offsets=np.asarray(offsets, np.int64), splits=np.asarray(splits, np.int16),
             labels=np.asarray(labels, np.int16), groups=np.asarray(groups, np.int64), meta=np.asarray(json.dumps(meta)))
    os.replace(pending, target)
    return target


@dataclass
class _Family:
    name: str
    path: Path
    identity: dict
    offsets: np.ndarray       # eligible records, grouped: records of group g are offsets[starts[g]:starts[g+1]]
    starts: np.ndarray
    weight: float


class PortRecordStream:
    def __init__(self, paths: dict[str, str | Path], renderer: Renderer, *, index_dir: str | Path, split: str = "train",
                 labels=("gold", "checked", "teacher"), weights: dict[str, float] | None = None, seed: int = 0,
                 natlang_share: float = 0.25, max_producer_tokens: int = 1536, max_target_tokens: int = 192,
                 max_consumer_tokens: int | None = None, held_log: str | Path | None = None):
        self.renderer, self.seed, self.natlang_share = renderer, seed, natlang_share
        self.bounds = (max_producer_tokens, max_target_tokens, max_consumer_tokens or max_producer_tokens)
        self.held_log = Path(held_log) if held_log else None
        self.families: list[_Family] = []
        for name in sorted(paths):
            path = Path(paths[name])
            with np.load(build_index(path, index_dir), allow_pickle=False) as index:
                meta = json.loads(str(index["meta"]))
                wanted = [code for label, code in meta["labels"].items() if label in set(labels)]
                keep = (index["splits"] == meta["splits"].get(split, -1)) & np.isin(index["labels"], wanted)
                offsets, groups = index["offsets"][keep], index["groups"][keep]
            if not len(offsets):
                raise ValueError(f"{name}: no {split} records with labels {sorted(labels)}")
            order = np.lexsort((offsets, groups))  # by group, file order within a group
            offsets, groups = offsets[order], groups[order]
            boundaries = np.flatnonzero(np.diff(groups)) + 1
            starts = np.concatenate([[0], boundaries, [len(offsets)]]).astype(np.int64)
            self.families.append(_Family(name, path, meta["identity"], offsets, starts,
                                         float((weights or {}).get(name, 1.0))))
        self.epochs = [0] * len(self.families)
        self.positions = [0] * len(self.families)
        self.credits = [0.0] * len(self.families)
        self.held = {f.name: 0 for f in self.families}
        self.served = {f.name: 0 for f in self.families}
        self._permutations: dict[tuple[int, int], np.ndarray] = {}

    # Sampling ------------------------------------------------------------------------------
    def _permutation(self, family: int, epoch: int) -> np.ndarray:
        key = (family, epoch)
        if key not in self._permutations:
            self._permutations.clear()  # one live permutation per family at a time is enough
            seed = _hash64(f"{self.seed}:{self.families[family].name}:{epoch}") & 0x7FFFFFFF
            self._permutations[key] = np.random.default_rng(seed).permutation(len(self.families[family].starts) - 1)
        return self._permutations[key]

    def _next_family(self) -> int:
        total = sum(f.weight for f in self.families)
        for i, f in enumerate(self.families):
            self.credits[i] += f.weight
        chosen = max(range(len(self.families)), key=lambda i: (self.credits[i], -i))
        self.credits[chosen] -= total
        return chosen

    def _next_offset(self, family: int) -> int:
        f = self.families[family]
        groups = len(f.starts) - 1
        if self.positions[family] >= groups:
            self.epochs[family] += 1
            self.positions[family] = 0
        group = int(self._permutation(family, self.epochs[family])[self.positions[family]])
        self.positions[family] += 1
        size = int(f.starts[group + 1] - f.starts[group])
        return int(f.offsets[f.starts[group] + self.epochs[family] % size])

    def _read(self, family: int, offset: int):
        f = self.families[family]
        if _identity(f.path) != f.identity:
            raise ValueError(f"{f.path} changed after it was indexed; start a new stream at a recorded boundary")
        with open(f.path, "rb") as handle:
            handle.seek(offset)
            return parse_record(json.loads(handle.readline()))

    def __iter__(self):
        return self

    def __next__(self) -> RenderedRecord:
        limit, skipped = 2 * sum(len(f.offsets) for f in self.families), 0
        while True:
            if skipped > limit:
                raise ValueError("every eligible record exceeds the length bounds; nothing to train on")
            family = self._next_family()
            record = self._read(family, self._next_offset(family))
            chosen = _hash64(f"{self.seed}:form:{record.id}") & 0xFFFF
            form = "natlang" if chosen < self.natlang_share * 0x10000 else "chat"
            rendered = render_record(self.renderer, record, form=form)
            producer, target, consumer = self.bounds
            if (len(rendered.producer) > producer or len(rendered.target) > target or
                    len(rendered.consumer_before) + len(rendered.consumer_after) > consumer):
                self.held[self.families[family].name] += 1
                if self.held_log:
                    with open(self.held_log, "a") as log:
                        log.write(json.dumps({"id": record.id, "family": self.families[family].name, "reason": "length",
                                              "producer": len(rendered.producer), "target": len(rendered.target)}) + "\n")
                skipped += 1
                continue
            self.served[self.families[family].name] += 1
            return rendered

    def take(self, n: int) -> list[RenderedRecord]:
        return [next(self) for _ in range(n)]

    # Resume ------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"version": "natlang.port-record-stream/1", "seed": self.seed, "natlang_share": self.natlang_share,
                "bounds": list(self.bounds),
                "families": [{"name": f.name, "identity": f.identity, "weight": f.weight, "records": int(len(f.offsets)),
                              "groups": int(len(f.starts) - 1)} for f in self.families],
                "epochs": list(self.epochs), "positions": list(self.positions), "credits": list(self.credits),
                "held": dict(self.held), "served": dict(self.served)}

    def load_state_dict(self, state: dict) -> None:
        mine = self.state_dict()
        for key in ("version", "seed", "natlang_share", "bounds", "families"):
            if state.get(key) != mine[key]:
                raise ValueError(f"record stream {key} differs from the checkpoint; refusing to resume onto other data")
        self.epochs, self.positions = list(state["epochs"]), list(state["positions"])
        self.credits, self.held, self.served = list(state["credits"]), dict(state["held"]), dict(state["served"])
