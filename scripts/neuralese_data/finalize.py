"""Streaming finalisation of a full port-record build (S1 §6): dedup, split closure, protected scan,
validation and a manifest, without holding records in memory.

`dedup.analyse` and `splits.close` work on in-memory record lists, which is right for samples.
A full build has millions of records whose sources can be long trajectories, so this module makes
three streaming passes over the converted files:

1. **Background pass.** Source-text digests and the split groups that share them, to find background
   material (texts shared by at least BACKGROUND_GROUPS groups; see dedup.py).
2. **Index pass.** Per record: split, group keys, example digest, question digest, own-source digest,
   a bottom-k MinHash sketch for document-like sources (consume, reconstruct and compare records whose
   own source is at most SKETCH_MAX_CHARS; trajectories rely on their task and question groups), the
   protected-set hit and a richness tuple. Records are addressed by their position in the stream.
3. **Write pass.** Exact example duplicates are dropped (the richest copy is kept), closure assigns a
   split per component (test > validation > train; protected hits force test), every kept record is
   validated, and records are written per family with a manifest of counts and rejections.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import time
import zlib
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from .dedup import BACKGROUND_GROUPS, BAND, MIN_SHINGLES, NEAR_THRESHOLD, SHINGLE, SKETCH, _one_source_text, consumer_text, target_text
from .records import normalize_text, validate
from .splits import RANK, protected_hit

SKETCH_MAX_CHARS = 60_000
SKETCH_TASKS = {"consume", "reconstruct", "compare"}


def _digest(text: str) -> bytes:
    return hashlib.blake2b(normalize_text(text).encode("utf-8"), digest_size=12).digest()


def _files(inputs: list[Path]) -> list[Path]:
    out = []
    for p in inputs:
        out.extend(sorted(p.glob("*.port-records.jsonl")) if p.is_dir() else [p])
    return out


def _stream(files):
    for path in files:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield path, json.loads(line)


def _sketch(text: str) -> array | None:
    words = normalize_text(text[:SKETCH_MAX_CHARS]).split(" ")
    if len(words) < SHINGLE + MIN_SHINGLES:
        return None
    hashes = {zlib.crc32(" ".join(words[i:i + SHINGLE]).encode("utf-8")) for i in range(len(words) - SHINGLE + 1)}
    return array("I", heapq.nsmallest(SKETCH, hashes))


def _jaccard(a: array, b: array) -> float:
    sa, sb = set(a), set(b)
    union = heapq.nsmallest(SKETCH, sa | sb)
    return sum(1 for h in union if h in sa and h in sb) / max(1, len(union))


class _UF:
    def __init__(self):
        self.parent = array("q")

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _richness(r: dict) -> tuple:
    return (len(r["target"].get("alternatives") or []), len((r.get("contrasts") or {}).get("distractors") or []),
            sum(len(s.get("exact_refs") or []) for s in r["sources"]))


def build(inputs: list[Path], out: Path, protected: dict | None, log=print) -> dict:
    files = _files(inputs)
    started = time.time()

    # Pass 1: background texts.
    first_group: dict[bytes, int] = {}
    multi: dict[bytes, set] = {}
    group_ids: dict[str, int] = {}
    for _, r in _stream(files):
        g = group_ids.setdefault(r["split_groups"][0], len(group_ids))
        for s in r["sources"]:
            d = _digest(_one_source_text(s))
            seen = first_group.setdefault(d, g)
            if seen != g:
                multi.setdefault(d, {seen}).add(g)
    background = {d for d, gs in multi.items() if len(gs) >= BACKGROUND_GROUPS}
    del first_group, multi, group_ids
    log(f"pass 1: {len(background)} background source texts ({time.time() - started:.0f}s)")

    # Pass 2: index.
    uf = _UF()
    group_node: dict[str, int] = {}
    split_of = bytearray()
    by_example: dict[bytes, int] = {}
    example_rich: dict[bytes, tuple] = {}
    drop: set = set()
    by_question: dict[bytes, int] = {}
    by_source: dict[bytes, int] = {}
    lsh: dict[int, list] = defaultdict(list)
    sketches: dict[int, array] = {}
    forced: list[int] = []
    counts = Counter()
    rec_nodes = array("q")  # record position → union-find node
    n = 0
    for _, r in _stream(files):
        node = uf.add()
        rec_nodes.append(node)
        split_of.append(RANK[r["split"]])
        for g in r["split_groups"]:
            gid = group_node.get(g)
            if gid is None:
                gid = group_node[g] = uf.add()
                split_of.append(0)
            uf.union(node, gid)
        src_texts = [_one_source_text(s) for s in r["sources"]]
        ctext, ttext = consumer_text(r), target_text(r)
        ex = _digest("\n\u0000".join(src_texts) + "\n\u0000" + ctext + "\n\u0000" + ttext)
        rich = _richness(r)
        if ex in by_example:
            counts["exact_example_duplicates"] += 1
            other = by_example[ex]
            uf.union(node, other)
            if rich > example_rich[ex]:
                drop.add(other)
                by_example[ex], example_rich[ex] = node, rich
            else:
                drop.add(node)
        else:
            by_example[ex], example_rich[ex] = node, rich
        q = _digest(ctext + "\n\u0000" + ttext)
        if q in by_question:
            uf.union(node, by_question[q])
            counts["same_question_links"] += 1
        else:
            by_question[q] = node
        own = [t for t in src_texts if _digest(t) not in background]
        if own:
            joined = "\n".join(own)
            sd = _digest(joined)
            if sd in by_source:
                uf.union(node, by_source[sd])
                counts["same_source_links"] += 1
            else:
                by_source[sd] = node
                if r["task"] in SKETCH_TASKS and len(joined) <= SKETCH_MAX_CHARS:
                    sk = _sketch(joined)
                    if sk is not None:
                        candidates = set()
                        for h in sk[:BAND]:
                            candidates.update(lsh[h])
                            lsh[h].append(node)
                        for other in candidates:
                            if _jaccard(sk, sketches[other]) >= NEAR_THRESHOLD:
                                uf.union(node, other)
                                counts["near_duplicate_source_links"] += 1
                        sketches[node] = sk
        if protected:
            hit = protected_hit(r, protected)
            if hit:
                counts[f"protected_{hit}"] += 1
                forced.append(node)
        n += 1
        if n % 200_000 == 0:
            log(f"pass 2: {n} records ({time.time() - started:.0f}s)")
    del by_example, example_rich, by_question, by_source, lsh, sketches
    log(f"pass 2: {n} records indexed, {len(drop)} duplicates to drop ({time.time() - started:.0f}s)")

    # Component splits.
    comp = {}
    for i in range(n):
        nd = rec_nodes[i]
        if nd in drop:
            continue
        root = uf.find(nd)
        if split_of[nd] > comp.get(root, 0):
            comp[root] = split_of[nd]
        else:
            comp.setdefault(root, split_of[nd])
    for nd in forced:
        comp[uf.find(nd)] = 2
    names = {v: k for k, v in RANK.items()}

    # Pass 3: write.
    out.mkdir(parents=True, exist_ok=True)
    writers: dict[str, object] = {}
    tmp_paths: dict[str, Path] = {}
    families: dict[str, Counter] = defaultdict(Counter)
    licences: dict[str, Counter] = defaultdict(Counter)
    moved = Counter()
    invalid = Counter()
    invalid_examples = {}
    group_split: dict[str, str] = {}
    violations = 0
    i = 0
    for _, r in _stream(files):
        nd = rec_nodes[i]
        i += 1
        if nd in drop:
            continue
        new = names[comp[uf.find(nd)]]
        if new != r["split"]:
            moved[f"{r['split']}->{new}"] += 1
            r["split"] = new
        errors = validate(r)
        if errors:
            key = errors[0].split(":")[0]
            invalid[key] += 1
            invalid_examples.setdefault(key, f"{r['id']}: {errors[0]}")
            continue
        for g in r["split_groups"]:
            if group_split.setdefault(g, new) != new:
                violations += 1
        fam = r["family"]
        if fam not in writers:
            tmp_paths[fam] = out / f"{fam}.port-records.jsonl.pending"
            writers[fam] = open(tmp_paths[fam], "w", encoding="utf-8")
        writers[fam].write(json.dumps(r, ensure_ascii=False) + "\n")
        families[fam][new] += 1
        licences[r["license"]["spdx"]][fam] += 1
    for fam, w in writers.items():
        w.close()
        os.replace(tmp_paths[fam], out / f"{fam}.port-records.jsonl")
    total = sum(sum(c.values()) for c in families.values())
    report = {
        "inputs": [str(p) for p in files],
        "records_in": n,
        "records_out": total,
        "dropped_exact_duplicates": len(drop),
        "invalid": dict(invalid),
        "invalid_examples": invalid_examples,
        "links": {k: v for k, v in counts.items() if not k.startswith("protected_")},
        "protected_hits": {k.removeprefix("protected_"): v for k, v in counts.items() if k.startswith("protected_")},
        "moved": dict(moved),
        "closure_violations": violations,
        "background_source_texts": len(background),
        "by_split": dict(sum((c for c in families.values()), Counter())),
        "families": {k: dict(v) for k, v in sorted(families.items())},
        "licences": {k: dict(v) for k, v in sorted(licences.items())},
        "seconds": round(time.time() - started),
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(f"pass 3: wrote {total} records in {len(families)} families ({report['seconds']}s)")
    return report

