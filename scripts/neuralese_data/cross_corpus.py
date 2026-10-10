"""Cross-corpus split closure and near-duplicate dedup against published corpora (S1 §6.4; VIEW_CORPUS.md §4).

A new corpus whose sources overlap a published one (shared `wiki:`/`repo:`/`sql-db:` groups, the same documents
under other group keys, the same questions) must not put a document in train while the published corpus has it in
test or validation, or the reverse. Published corpora are immutable, so the new corpus adapts:

1. **Index** (`index_corpus`, once per published corpus, persisted): one streaming pass, parallel over byte ranges,
   records per published record its split under every split-group key, the digest of each source text, of the
   question (consumer request + target) and of the whole example, and a bottom-k MinHash sketch of each source text
   (the finalizer's digest and sketch functions; consume/reconstruct/compare records, sources up to 60k characters).
   Source texts that occur under three or more groups are background material (tool documentation, rules) and are
   not used as document identity, as in the finalizer.
2. **Apply** (`apply`): the new records are closed internally (their groups plus the dedup links: same source,
   same question, near-duplicate source, computed once per distinct source text; exact example duplicates dropped). Each component then collects the published splits it touches: through a
   shared group key, an identical non-background source text, an identical question, or a near-duplicate source
   (estimated Jaccard ≥ 0.8). A record whose whole example (sources, request, target) is already published is
   dropped. A component that touches one published split takes it, unless its own upstream split (or a protected
   benchmark hit) ranks higher: then, and when it touches several published splits, the component is dropped. The
   published corpus cannot move, and a test document never moves to train.
3. **Check** (`check`): no kept record shares a group, a non-background source or a question with a published record
   of another split.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from array import array
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from .dedup import BAND, NEAR_THRESHOLD, SKETCH, _one_source_text, consumer_text, target_text
from .finalize import SKETCH_MAX_CHARS, SKETCH_TASKS, _digest, _sketch
from .splits import RANK, protected_hit

SCHEMA = "natlang.cross-corpus-index/1"
NAMES = {v: k for k, v in RANK.items()}
BACKGROUND = 3
CHUNK = 256 << 20


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------------------------------- index

def _chunks(files: list[Path]) -> list[tuple[str, int, int]]:
    out = []
    for f in files:
        size = f.stat().st_size
        for start in range(0, size, CHUNK):
            out.append((str(f), start, min(size, start + CHUNK)))
    return out


def _port_view(r: dict) -> tuple[str, list[str], list[str], str, str, str]:
    texts = [_one_source_text(s) for s in r["sources"]]
    return r["split"], r["split_groups"], texts, r["task"], consumer_text(r), target_text(r)


def _turn_view(r: dict) -> tuple[str, list[str], list[str], str, str, str]:
    """natlang.teacher_training_turn records (harness bench): groups only; their sources are trajectories."""
    return r["split"], r.get("source_groups") or [], [], "continue", "", ""


def _index_chunk(job):
    path, start, end, fmt = job
    view = _port_view if fmt == "port" else _turn_view
    groups: dict[str, int] = {}
    group_conflicts: list = []
    sources: dict[bytes, list] = {}   # digest -> [min rank, max rank, first groups (≤ BACKGROUND)]
    questions: dict[bytes, list] = {}
    examples: dict[bytes, int] = {}
    sketches: dict[bytes, array] = {}
    n = 0
    with open(path, "rb") as stream:
        if start:
            stream.seek(start - 1)
            if stream.read(1) != b"\n":
                stream.readline()  # the line straddling `start` belongs to the previous chunk
        while stream.tell() < end:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            split, gs, texts, task, ctext, ttext = view(json.loads(line))
            rank = RANK[split]
            n += 1
            for g in gs:
                old = groups.setdefault(g, rank)
                if old != rank:
                    group_conflicts.append((g, NAMES[old], split))
                    groups[g] = max(old, rank)
            first = gs[0] if gs else ""
            for t in texts:
                d = _digest(t)
                e = sources.get(d)
                if e is None:
                    sources[d] = [rank, rank, {first}]
                    if task in SKETCH_TASKS and len(t) <= SKETCH_MAX_CHARS:
                        sk = _sketch(t)
                        if sk is not None:
                            sketches[d] = sk
                else:
                    e[0], e[1] = min(e[0], rank), max(e[1], rank)
                    if len(e[2]) < BACKGROUND:
                        e[2].add(first)
            if ctext or ttext:
                q = _digest(ctext + "\n\u0000" + ttext)
                e = questions.get(q)
                questions[q] = [rank, rank] if e is None else [min(e[0], rank), max(e[1], rank)]
                ex = _digest("\n\u0000".join(texts) + "\n\u0000" + ctext + "\n\u0000" + ttext)
                examples[ex] = max(examples.get(ex, 0), rank)
    return {"n": n, "groups": groups, "group_conflicts": group_conflicts[:100], "sources": sources,
            "questions": questions, "examples": examples, "sketches": sketches}


def index_corpus(files: list[Path], out: Path, *, corpus_id: str, fmt: str = "port", workers: int = 4,
                 log=print) -> dict:
    """Index a published corpus into `out` (groups.json, digests.npz, sketches.npz, index.json)."""
    import numpy as np

    started = time.time()
    jobs = [(p, a, b, fmt) for p, a, b in _chunks(files)]
    groups: dict[str, int] = {}
    conflicts: list = []
    sources: dict[bytes, list] = {}
    questions: dict[bytes, list] = {}
    examples: dict[bytes, int] = {}
    sketches: dict[bytes, array] = {}
    records = 0
    with Pool(workers) as pool:
        for k, part in enumerate(pool.imap_unordered(_index_chunk, jobs)):
            records += part["n"]
            for g, rank in part["groups"].items():
                old = groups.setdefault(g, rank)
                if old != rank:
                    conflicts.append((g, NAMES[old], NAMES[rank]))
                    groups[g] = max(old, rank)
            conflicts.extend(part["group_conflicts"])
            for d, (lo, hi, gs) in part["sources"].items():
                e = sources.get(d)
                if e is None:
                    sources[d] = [lo, hi, gs]
                else:
                    e[0], e[1] = min(e[0], lo), max(e[1], hi)
                    if len(e[2]) < BACKGROUND:
                        e[2] |= set(list(gs)[:BACKGROUND])
            for q, (lo, hi) in part["questions"].items():
                e = questions.get(q)
                questions[q] = [lo, hi] if e is None else [min(e[0], lo), max(e[1], hi)]
            for ex, rank in part["examples"].items():
                examples[ex] = max(examples.get(ex, 0), rank)
            for d, sk in part["sketches"].items():
                sketches.setdefault(d, sk)
            if k % 20 == 0:
                log(f"index {corpus_id}: {k + 1}/{len(jobs)} chunks, {records} records ({time.time() - started:.0f}s)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "groups.json").write_text(json.dumps({g: NAMES[r] for g, r in sorted(groups.items())}) + "\n")

    def packed(d: dict, value):
        keys = sorted(d)  # hex keys: numpy byte strings drop trailing NULs, hex has none
        return (np.array([k.hex() for k in keys], dtype="S24"), np.array([value(d[k]) for k in keys], dtype=np.uint8))

    # Source value: min rank | max rank << 2 | background << 4.
    src_keys, src_vals = packed(sources, lambda e: e[0] | (e[1] << 2) | ((len(e[2]) >= BACKGROUND) << 4))
    q_keys, q_vals = packed(questions, lambda e: e[0] | (e[1] << 2))
    ex_keys, ex_vals = packed(examples, lambda e: e)
    np.savez(out / "digests.npz", source_keys=src_keys, source_values=src_vals, question_keys=q_keys,
             question_values=q_vals, example_keys=ex_keys, example_values=ex_vals)
    sk_keys = sorted(sketches)
    matrix = np.zeros((len(sk_keys), SKETCH), dtype=np.uint32)
    for i, d in enumerate(sk_keys):
        s = sketches[d]
        matrix[i, :len(s)] = s
        matrix[i, len(s):] = 0xFFFFFFFF  # padding never matches a crc32 bottom-k entry in practice
    lengths = np.array([len(sketches[d]) for d in sk_keys], dtype=np.uint8)
    np.savez(out / "sketches.npz", keys=np.array([k.hex() for k in sk_keys], dtype="S24"), sketches=matrix, lengths=lengths)
    meta = {"schema": SCHEMA, "corpus_id": corpus_id, "format": fmt, "files": [str(f) for f in files],
            "file_bytes": {str(f): f.stat().st_size for f in files}, "records": records, "groups": len(groups),
            "group_conflicts": len(conflicts), "group_conflict_examples": conflicts[:20], "sources": len(sources),
            "background_sources": int(sum(1 for e in sources.values() if len(e[2]) >= BACKGROUND)),
            "questions": len(questions), "examples": len(examples), "sketches": len(sk_keys),
            "seconds": round(time.time() - started), "sketch": {"shingle": 5, "k": SKETCH, "band": BAND,
                                                                "max_chars": SKETCH_MAX_CHARS, "tasks": sorted(SKETCH_TASKS)}}
    (out / "index.json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"index {corpus_id}: {json.dumps({k: v for k, v in meta.items() if k not in ('files', 'file_bytes')})}")
    return meta


class Index:
    """A loaded published-corpus index."""

    def __init__(self, root: Path):
        import numpy as np

        self.root = Path(root)
        self.meta = json.loads((self.root / "index.json").read_text())
        self.corpus_id = self.meta["corpus_id"]
        self.groups = {g: RANK[s] for g, s in json.loads((self.root / "groups.json").read_text()).items()}
        dg = np.load(self.root / "digests.npz")

        def hexed(keys):
            # Early indexes stored raw 12-byte digests as S12; the buffer keeps trailing NULs, element access does not.
            if keys.dtype == np.dtype("S12"):
                raw = keys.tobytes()
                return np.array([raw[i:i + 12].hex() for i in range(0, len(raw), 12)], dtype="S24")
            return keys

        self._src = (hexed(dg["source_keys"]), dg["source_values"])
        self._q = (hexed(dg["question_keys"]), dg["question_values"])
        self._ex = (hexed(dg["example_keys"]), dg["example_values"])
        sk = np.load(self.root / "sketches.npz")
        self.sketch_keys, self.sketches, self.sketch_lengths = hexed(sk["keys"]), sk["sketches"], sk["lengths"]

    @staticmethod
    def _lookup(table, digest: bytes):
        import numpy as np

        keys, values = table
        if not len(keys):
            return None
        key = digest.hex().encode()
        i = int(np.searchsorted(keys, np.array([key], dtype="S24"))[0])
        if i < len(keys) and keys[i] == key:
            return int(values[i])
        return None

    def source(self, text: str) -> tuple[int, int] | None:
        """(min rank, max rank) of a non-background published source with this text, else None."""
        v = self._lookup(self._src, _digest(text))
        if v is None or v & 16:
            return None
        return v & 3, (v >> 2) & 3

    def source_value(self, digest: bytes):
        return self._lookup(self._src, digest)

    def question(self, ctext: str, ttext: str) -> tuple[int, int] | None:
        v = self._lookup(self._q, _digest(ctext + "\n\u0000" + ttext))
        return None if v is None else (v & 3, (v >> 2) & 3)

    def example(self, texts: list[str], ctext: str, ttext: str) -> int | None:
        return self._lookup(self._ex, _digest("\n\u0000".join(texts) + "\n\u0000" + ctext + "\n\u0000" + ttext))

    def near(self, sketches: dict[str, array]) -> dict[str, list[tuple[int, int, float]]]:
        """Near-duplicate published sources for our sketches: {our key: [(min rank, max rank, jaccard)]}.
        LSH over our (few) sketches; the published sketch matrix is streamed against it."""
        band: dict[int, list[str]] = defaultdict(list)
        for key, sk in sketches.items():
            for h in list(sk)[:BAND]:
                band[h].append(key)
        out: dict[str, list] = defaultdict(list)
        ours = {k: set(v) for k, v in sketches.items()}
        for i in range(len(self.sketch_keys)):
            n = int(self.sketch_lengths[i])
            row = self.sketches[i, :n]
            cands = set()
            for h in row[:BAND].tolist():
                if h in band:
                    cands.update(band[h])
            if not cands:
                continue
            theirs = set(row.tolist())
            for key in cands:
                mine = ours[key]
                union = sorted(mine | theirs)[:SKETCH]
                j = sum(1 for h in union if h in mine and h in theirs) / max(1, len(union))
                if j >= NEAR_THRESHOLD:
                    v = self.source_value(bytes.fromhex(self.sketch_keys[i].decode()))
                    if v is not None and not v & 16:
                        out[key].append((v & 3, (v >> 2) & 3, round(j, 3)))
        return out


# ---------------------------------------------------------------------------------------------- apply

def has_question(r: dict) -> bool:
    """Whether a record's request identifies it. Templated requests (code-computed extraction such as "What exit
    code did the command finish with?", reconstruction) recur over unrelated sources with equal targets, so a
    question link would chain them; only dataset questions, queries and docstring summaries are linked."""
    notes = r["lineage"].get("notes") or {}
    if r["task"] == "reconstruct":
        return False
    return "question" in notes or notes.get("target_origin") in ("dataset-summary", "upstream-docstring") or \
        notes.get("target_origin") is None

def _text(r: dict) -> str:
    return "\n".join(_one_source_text(s) for s in r["sources"])


def apply(records: list[dict], indexes: list[Index], *, protected: dict | None = None, log=print) -> tuple[list[dict], dict]:
    """Close `records` internally and against the published `indexes`; return (kept records, report).
    Record splits are assigned in place. Internal links are the dedup ones, computed once per distinct source text:
    same source, same question (request + target), near-duplicate source; exact example duplicates keep the first."""
    report: dict = {"records_in": len(records), "indexes": [ix.corpus_id for ix in indexes]}
    uf = _UF()
    drop: set = set()
    by_example: dict[bytes, str] = {}
    by_question: dict[bytes, str] = {}
    by_source: dict[bytes, str] = {}
    src_of: dict[str, bytes] = {}
    sketches: dict[str, array] = {}
    counts = Counter()
    for r in records:
        rid = r["id"]
        for g in r["split_groups"]:
            uf.union("r:" + rid, "g:" + g)
        texts = [_one_source_text(s) for s in r["sources"]]
        ctext, ttext = consumer_text(r), target_text(r)
        ex = _digest("\n\u0000".join(texts) + "\n\u0000" + ctext + "\n\u0000" + ttext)
        if ex in by_example:
            drop.add(rid)
            counts["internal_exact_example_duplicates"] += 1
            continue
        by_example[ex] = rid
        q = _digest(ctext + "\n\u0000" + ttext)
        if not has_question(r):
            pass
        elif q in by_question:
            uf.union("r:" + rid, "r:" + by_question[q])
            counts["internal_same_question_links"] += 1
        else:
            by_question[q] = rid
        joined = "\n".join(texts)
        sd = _digest(joined)
        src_of[rid] = sd
        if sd in by_source:
            uf.union("r:" + rid, "r:" + by_source[sd])
        else:
            by_source[sd] = rid
            if len(joined) <= SKETCH_MAX_CHARS:
                sk = _sketch(joined)
                if sk is not None:
                    sketches[sd.hex()] = sk
    # Internal near duplicates over distinct sources (LSH on the first BAND hashes, as the finalizer).
    lsh: dict[int, list[str]] = defaultdict(list)
    for key, sk in sketches.items():
        cands = set()
        for h in list(sk)[:BAND]:
            cands.update(lsh[h])
            lsh[h].append(key)
        mine = set(sk)
        for other in cands:
            theirs = set(sketches[other])
            union = sorted(mine | theirs)[:SKETCH]
            if sum(1 for h in union if h in mine and h in theirs) / max(1, len(union)) >= NEAR_THRESHOLD:
                uf.union("r:" + by_source[bytes.fromhex(key)], "r:" + by_source[bytes.fromhex(other)])
                counts["internal_near_duplicate_source_links"] += 1
    del lsh
    # External constraints per record.
    ext: dict[str, set] = defaultdict(set)   # record id -> {(corpus, rank, via)}
    reasons = Counter()
    published_dups = set()
    for r in records:
        if r["id"] in drop:
            continue
        texts = [_one_source_text(s) for s in r["sources"]]
        ctext, ttext = consumer_text(r), target_text(r)
        for ix in indexes:
            for g in r["split_groups"]:
                if g in ix.groups:
                    ext[r["id"]].add((ix.corpus_id, ix.groups[g], "group"))
                    reasons[f"{ix.corpus_id}:group"] += 1
            for t in texts:
                hit = ix.source(t)
                if hit:
                    for rank in set(hit):
                        ext[r["id"]].add((ix.corpus_id, rank, "source"))
                    reasons[f"{ix.corpus_id}:same-source"] += 1
            q = ix.question(ctext, ttext) if has_question(r) else None
            if q:
                for rank in set(q):
                    ext[r["id"]].add((ix.corpus_id, rank, "question"))
                reasons[f"{ix.corpus_id}:same-question"] += 1
            if ix.example(texts, ctext, ttext) is not None:
                published_dups.add(r["id"])
                reasons[f"{ix.corpus_id}:exact-example"] += 1
    by_digest: dict[str, list[str]] = defaultdict(list)
    for rid, sd in src_of.items():
        if rid not in drop:
            by_digest[sd.hex()].append(rid)
    for ix in indexes:
        for d, hits in ix.near(sketches).items():
            for lo, hi, _j in hits:
                for rid in by_digest[d]:
                    ext[rid].update({(ix.corpus_id, lo, "near"), (ix.corpus_id, hi, "near")})
            reasons[f"{ix.corpus_id}:near-duplicate-source"] += len(by_digest[d])
    # Components.
    comps: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["id"] not in drop:
            comps[uf.find("r:" + r["id"])].append(r)
    kept, moved, dropped = [], Counter(), Counter()
    protected_hits = Counter()
    for members in comps.values():
        own = max(RANK[r["split"]] for r in members)
        forced = False
        if protected:
            for r in members:
                hit = protected_hit(r, protected)
                if hit:
                    protected_hits[hit] += 1
                    forced = True
        if forced:
            own = RANK["test"]
        touched = {rank for r in members for (_c, rank, _v) in ext.get(r["id"], ())}
        if not touched:
            target = own
        elif len(touched) == 1 and own <= next(iter(touched)):
            target = next(iter(touched))
        else:
            why = ("several published splits " + "/".join(NAMES[t] for t in sorted(touched))) if len(touched) > 1 else \
                f"own {NAMES[own]}{' (protected)' if forced else ''} vs published {NAMES[next(iter(touched))]}"
            for r in members:
                dropped[f"{r['lineage']['store']}: {why}"] += 1
            continue
        for r in members:
            if r["id"] in published_dups:
                dropped[f"{r['lineage']['store']}: exact example already published"] += 1
                continue
            new = NAMES[target]
            if new != r["split"]:
                moved[f"{r['lineage']['store']}:{r['split']}->{new}"] += 1
                r["split"] = new
            kept.append(r)
    report.update({"internal": dict(counts), "external_links": dict(sorted(reasons.items())),
                   "protected_hits": dict(protected_hits), "components": len(comps),
                   "moved": dict(sorted(moved.items())), "dropped": dict(sorted(dropped.items())),
                   "records_out": len(kept), "by_split": dict(Counter(r["split"] for r in kept))})
    log(f"cross-corpus: {json.dumps({k: report[k] for k in ('records_in', 'records_out', 'components')})}")
    return kept, report


def check(records: list[dict], indexes: list[Index]) -> list[str]:
    """Violations: a kept record that shares a group, a non-background source or a question with a published record
    of another split."""
    errors = []
    for r in records:
        rank = RANK[r["split"]]
        texts = [_one_source_text(s) for s in r["sources"]]
        for ix in indexes:
            for g in r["split_groups"]:
                if g in ix.groups and ix.groups[g] != rank:
                    errors.append(f"{r['id']}: group {g} is {NAMES[ix.groups[g]]} in {ix.corpus_id}")
            for t in texts:
                hit = ix.source(t)
                if hit and set(hit) != {rank}:
                    errors.append(f"{r['id']}: source in {ix.corpus_id} splits {[NAMES[h] for h in set(hit)]}")
            q = ix.question(consumer_text(r), target_text(r)) if has_question(r) else None
            if q and set(q) != {rank}:
                errors.append(f"{r['id']}: question in {ix.corpus_id} splits {[NAMES[h] for h in set(q)]}")
    return errors


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Index a published corpus for cross-corpus closure.")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--format", choices=("port", "turn"), default="port")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    files = []
    for f in args.files:
        f = f.resolve()
        files.extend(sorted(f.glob("*.port-records.jsonl")) if f.is_dir() else [f])
    index_corpus(files, args.out, corpus_id=args.corpus_id, fmt=args.format, workers=args.workers,
                 log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
