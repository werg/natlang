#!/usr/bin/env python3
"""The S3 full run's data (plans/neuralese/S3_FULL_RUN_PLAN.md §2) as a subset on NVMe.

Per family of the audited S1 final corpus: at most `--train-per-family` training records, chosen as the training
stream samples them (one record per source group per round, groups in a seeded permutation, file order within a
group), so no large group of near-duplicates dominates; and the first `--eval-rows` evaluation records (validation or
test, file order; the harness takes its 64 that fit the length bounds from these). Only the listed outcome labels.
Records keep their bytes and file order. The stream then reads the subset with fast random access instead of
seeking on the external disk. A manifest records the source identity, the selection and the counts.

    python3 scripts/neuralese_s3_subset.py --source FINAL_DIR --out /home/werg/data/neuralese-s3-full/records
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "neuralese"))
from natlang_neuralese.data.stream import build_index  # noqa: E402

FINAL = "/mnt/external/natlang-development-data/data/neuralese/port-records/full-20261003/resume-20261003T1948-review1/final"


def select_train(groups: np.ndarray, offsets: np.ndarray, cap: int, seed: str) -> np.ndarray:
    """Up to `cap` offsets: round-robin over source groups in a seeded permutation, file order within a group."""
    order = np.lexsort((offsets, groups))
    offsets, groups = offsets[order], groups[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(groups)) + 1, [len(offsets)]])
    sizes = np.diff(starts)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little"))
    permutation = rng.permutation(len(sizes))
    chosen, round_ = [], 0
    while len(chosen) < cap:
        live = permutation[sizes[permutation] > round_]
        if not len(live):
            break
        take = live[: cap - len(chosen)]
        chosen.extend(offsets[starts[take] + round_].tolist())
        round_ += 1
    return np.sort(np.asarray(chosen, np.int64))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default=FINAL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-per-family", type=int, default=25000)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--labels", default="gold,checked,teacher")
    parser.add_argument("--seed", default="s3-full-20261005")
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_dir = out.parent / "source-index"
    labels = set(args.labels.split(","))
    manifest = {"schema": "natlang.s3-full-subset/1", "source": str(source), "seed": args.seed,
                "train_per_family": args.train_per_family, "eval_rows": args.eval_rows, "labels": sorted(labels),
                "families": {}}
    for path in sorted(source.glob("*.port-records.jsonl")):
        family = path.name.removesuffix(".port-records.jsonl")
        target = out / path.name
        started = time.time()
        with np.load(build_index(path, index_dir), allow_pickle=False) as index:
            meta = json.loads(str(index["meta"]))
            wanted = [code for label, code in meta["labels"].items() if label in labels]
            eligible = np.isin(index["labels"], wanted)
            train = eligible & (index["splits"] == meta["splits"].get("train", -1))
            held = eligible & np.isin(index["splits"], [meta["splits"][s] for s in ("validation", "test") if s in meta["splits"]])
            chosen = select_train(index["groups"][train], index["offsets"][train], args.train_per_family, f"{args.seed}:{family}")
            evaluation = np.sort(index["offsets"][held])[: args.eval_rows]
        keep = np.unique(np.concatenate([chosen, evaluation]))
        pending = target.with_suffix(".pending")
        with open(path, "rb") as src, open(pending, "wb") as dst:
            for offset in keep.tolist():
                src.seek(offset)
                dst.write(src.readline())
        pending.replace(target)
        manifest["families"][family] = {"source_identity": meta["identity"], "train": int(len(chosen)),
                                        "eval": int(len(evaluation)), "train_available": int(train.sum()),
                                        "bytes": target.stat().st_size, "seconds": round(time.time() - started, 1)}
        print(json.dumps({family: manifest["families"][family]}), flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
