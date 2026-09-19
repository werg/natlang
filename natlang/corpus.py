"""Corpus identity and program-level splits, shared by training and evaluation."""
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def records(path):
    path = Path(path)
    files = sorted(path.glob("part-*.jsonl.gz")) if path.is_dir() else [path]
    for file in files:
        with (gzip.open(file, "rt") if file.suffix == ".gz" else file.open()) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def program_id(row):
    if row.get("program_id"):
        return row["program_id"]
    # Legacy generator IDs: <family>-<program index>-<turn index>.
    name, sep, turn = row["id"].rpartition("-")
    if not sep or not turn.isdigit():
        raise ValueError(f"Cannot recover program identity from {row['id']!r}")
    return name


def split_programs(pairs, holdout=200, seed=0):
    """Hold out whole programs until at least `holdout` turns are reserved."""
    if holdout < 0:
        raise ValueError("holdout must be nonnegative")
    groups = defaultdict(list)
    for row in pairs:
        groups[program_id(row)].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    held_keys, count = set(), 0
    for key in keys:
        if count >= holdout:
            break
        held_keys.add(key)
        count += len(groups[key])
    held = [p for p in pairs if program_id(p) in held_keys]
    train = [p for p in pairs if program_id(p) not in held_keys]
    if not train:
        raise ValueError("No training programs remain; reduce --holdout or supply more programs")
    rng.shuffle(held)
    rng.shuffle(train)
    manifest = {"version": "program-split/1", "seed": seed, "holdout_target": holdout,
                "held_programs": sorted(held_keys), "train_programs": sorted(set(keys) - held_keys),
                "held_turns": len(held), "train_turns": len(train)}
    return held, train, manifest


def index_pairs(path):
    """Index SFT rows without retaining every rendered conversation in RAM."""
    rows = []
    with Path(path).open('rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({'id': row['id'], 'program_id': program_id(row), 'offset': offset})
    return rows
