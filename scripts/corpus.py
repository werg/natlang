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
    """Split whole linked program groups, honoring explicit source splits."""
    if holdout < 0:
        raise ValueError("holdout must be nonnegative")
    groups = defaultdict(list)
    for row in pairs:
        groups[program_id(row)].append(row)
    # source_groups links derived/captured records that must never cross the split.
    parent = {key: key for key in groups}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def join(left, right):
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    owners = defaultdict(list)
    for key, rows in groups.items():
        for row in rows:
            for group in row.get("source_groups", []) or []:
                owners[str(group)].append(key)
    for keys_for_group in owners.values():
        for key in keys_for_group[1:]:
            join(keys_for_group[0], key)
    components = defaultdict(list)
    for key in groups:
        components[root(key)].extend(groups[key])
    explicit = defaultdict(set)
    for row in pairs:
        split = row.get("split")
        if split in ("train", "test"):
            explicit[root(program_id(row))].add(split)
        elif split is not None:
            raise ValueError(f"Unsupported explicit corpus split {split!r}")
    if any(len(labels) > 1 for labels in explicit.values()):
        raise ValueError("Linked corpus records have conflicting explicit train/test splits")
    fixed_held = {key for key, labels in explicit.items() if "test" in labels}
    fixed_train = {key for key, labels in explicit.items() if "train" in labels}
    if fixed_held & fixed_train:
        raise ValueError("Linked corpus records have conflicting explicit train/test splits")
    eligible = sorted(set(components) - fixed_held - fixed_train)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    held_keys, count = set(fixed_held), sum(len(components[k]) for k in fixed_held)
    for key in eligible:
        if count >= holdout:
            break
        held_keys.add(key)
        count += len(components[key])
    held_rows = {id(p) for key in held_keys for p in components[key]}
    held = [p for p in pairs if id(p) in held_rows]
    train = [p for p in pairs if id(p) not in held_rows]
    if not train:
        raise ValueError("No training programs remain; reduce --holdout or supply more programs")
    rng.shuffle(held)
    rng.shuffle(train)
    manifest = {"version": "program-split/1", "seed": seed, "holdout_target": holdout,
                "held_programs": sorted(k for k in groups if root(k) in held_keys),
                "train_programs": sorted(k for k in groups if root(k) not in held_keys),
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
            indexed = {'id': row['id'], 'program_id': program_id(row), 'offset': offset}
            if 'split' in row:
                indexed['split'] = row['split']
            if 'source_groups' in row:
                indexed['source_groups'] = row['source_groups']
            rows.append(indexed)
    return rows
