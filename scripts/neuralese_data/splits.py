"""Split-group closure and protected held-out scans for port records (S1 §6.2, §6.3).

Closure is a union-find over split-group keys. Each record joins all of its keys; dedup links
(`dedup.links.jsonl`) and protected-set hits add further joins. A component's split is the most
protected split of any member (test > validation > train): nothing moves from test to train, and
upstream held-out data pulls its whole component with it. Records whose split changed are counted.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .dedup import consumer_text
from .records import text_hash

RANK = {"train": 0, "validation": 1, "test": 2}
BENCHMARKS = Path("/home/werg/bgkit-data-nvme/bgkit2/benchmarks")
QUESTION_KEYS = ("question", "query", "problem_statement")
ID_KEYS = ("id", "instance_id", "query_id")


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _rows_from_file(path: Path):
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".arrow":
        import pyarrow as pa  # HF datasets arrow files are IPC streams

        with pa.ipc.open_stream(str(path)) as reader:
            for batch in reader:
                for start in range(0, batch.num_rows, 128):
                    yield from batch.slice(start, 128).to_pylist()
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        with pq.ParquetFile(str(path)) as reader:
            for batch in reader.iter_batches(batch_size=128):
                yield from batch.to_pylist()


def build_protected(paths: list[Path]) -> dict:
    """Normalised question hashes and ids of protected evaluation items."""
    hashes: dict[str, str] = {}
    ids: dict[str, str] = {}
    for path in paths:
        for row in _rows_from_file(path):
            for key in QUESTION_KEYS:
                if isinstance(row.get(key), str) and row[key].strip():
                    hashes.setdefault(text_hash(row[key]), str(path))
            for key in ID_KEYS:
                if row.get(key) is not None:
                    ids.setdefault(str(row[key]).split(":", 1)[-1], str(path))
    return {"question_hashes": hashes, "ids": ids}


def default_protected_paths(root: Path = BENCHMARKS) -> list[Path]:
    """bgkit benchmark items that are held out: showcase and web sets, SWE-bench Lite, browsecomp,
    and the validation splits of SQuAD, HotpotQA (distractor) and NQ-open."""
    paths = sorted((root / "showcase").glob("*.jsonl")) + sorted((root / "web").glob("*.jsonl"))
    paths += [root / "swebench_lite" / "instances.jsonl", root / "browsecomp_plus" / "queries.jsonl"]
    for name in ("squad", "hotpotqa_distractor", "nq_open"):
        paths += sorted((root / name / "validation").glob("*.arrow"))
    return [p for p in paths if p.exists()]


def protected_hit(record: dict, protected: dict) -> str | None:
    question = consumer_text(record)
    if question and text_hash(question) in protected["question_hashes"]:
        return "question"
    upstream_id = (record.get("lineage") or {}).get("upstream_id")
    if upstream_id and str(upstream_id) in protected["ids"]:
        return "id"
    notes = (record.get("lineage") or {}).get("notes") or {}
    instance = notes.get("instance_id")
    if instance and instance in protected["ids"]:
        return "instance"
    # Continuation records embed the question in a task statement; converters record it in notes.
    note_question = notes.get("question")
    if isinstance(note_question, str) and note_question.strip() and text_hash(note_question) in protected["question_hashes"]:
        return "question_note"
    return None


def close(records: list[dict], links: list[tuple[str, str]] = (), protected: dict | None = None) -> dict:
    """Assign closed splits in place; return a report."""
    uf = UnionFind()
    by_id = {r["id"]: r for r in records}
    for r in records:
        node = "record:" + r["id"]
        for g in r["split_groups"]:
            uf.union(node, "group:" + g)
    for a, b in links:
        if a in by_id and b in by_id:
            uf.union("record:" + a, "record:" + b)
    hits = Counter()
    forced = set()
    if protected:
        for r in records:
            hit = protected_hit(r, protected)
            if hit:
                hits[hit] += 1
                forced.add(uf.find("record:" + r["id"]))
    component_split: dict[str, str] = {}
    sizes: Counter = Counter()
    for r in records:
        root = uf.find("record:" + r["id"])
        sizes[root] += 1
        best = component_split.get(root, "train")
        if RANK[r["split"]] > RANK[best]:
            best = r["split"]
        component_split[root] = best
    for root in forced:
        component_split[root] = "test"
    moved = Counter()
    for r in records:
        new = component_split[uf.find("record:" + r["id"])]
        if new != r["split"]:
            moved[f"{r['split']}->{new}"] += 1
            r["split"] = new
    by_split = Counter(r["split"] for r in records)
    largest = sizes.most_common(3)
    return {
        "records": len(records),
        "components": len(sizes),
        "largest_components": [n for _, n in largest],
        "moved": dict(moved),
        "protected_hits": dict(hits),
        "by_split": dict(by_split),
    }


def check_closed(records: list[dict]) -> list[str]:
    """No split-group key may appear in two splits."""
    seen: dict[str, str] = {}
    errors = []
    for r in records:
        for g in r["split_groups"]:
            if seen.setdefault(g, r["split"]) != r["split"]:
                errors.append(f"{g}: {seen[g]} and {r['split']} ({r['id']})")
    return errors


def family_counts(records: list[dict]) -> dict:
    out: dict = defaultdict(Counter)
    for r in records:
        out[r["family"]][r["split"]] += 1
    return {k: dict(v) for k, v in sorted(out.items())}
