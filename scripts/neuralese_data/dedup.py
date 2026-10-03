"""Cross-corpus deduplication for port records (S1 §6.4).

Three levels, all stdlib:

* **Exact example duplicates** — same normalised sources, consumer request and target. All but one
  copy are dropped; the kept copy is the richer one (more alternatives, distractors, refs).
* **Question duplicates** — same consumer request and target over different sources (for example a
  bgkit question asked over a full article and over a window). Kept, but linked for split closure.
* **Near-duplicate sources** — bottom-k MinHash over word 5-shingles of the source text, candidates
  from the k_band smallest hashes, Jaccard estimated from the bottom-k union. Kept, linked for
  closure, so the same document cannot sit in two splits under different group keys.

Background material — a source text shared verbatim by records of at least BACKGROUND_GROUPS
different split groups (rules, worked examples, tool documentation, protocols) — is shared by
construction, not a document identity. It is left out of the same-source and near-duplicate
comparison so that it does not chain unrelated tasks into one split.
"""
from __future__ import annotations

import heapq
import json
import zlib
from collections import defaultdict
from pathlib import Path

from .records import canonical_json, normalize_text, text_hash

SHINGLE = 5
SKETCH = 64
BAND = 8
NEAR_THRESHOLD = 0.8
BACKGROUND_GROUPS = 3
MIN_SHINGLES = 20


def _one_source_text(s: dict) -> str:
    return s["text"] if "text" in s else "\n".join(m.get("content") or "" for m in s["messages"])


def source_text(record: dict, skip: set | None = None) -> str:
    parts = []
    for s in record["sources"]:
        text = _one_source_text(s)
        if skip and text_hash(text) in skip:
            continue
        parts.append(text)
    return "\n".join(parts)


def background_sources(records: list[dict]) -> set:
    """Hashes of source texts shared across at least BACKGROUND_GROUPS split groups."""
    groups: dict[str, set] = defaultdict(set)
    for r in records:
        key = r["split_groups"][0]
        for s in r["sources"]:
            groups[text_hash(_one_source_text(s))].add(key)
    return {h for h, g in groups.items() if len(g) >= BACKGROUND_GROUPS}


def consumer_text(record: dict) -> str:
    return "\n".join(m.get("content") or "" for m in record["consumer"]["context"] if m.get("role") == "user")


def target_text(record: dict) -> str:
    value = record["target"]["value"]
    return value if isinstance(value, str) else canonical_json(value)


def sketch(text: str) -> list[int]:
    words = normalize_text(text).split(" ")
    if len(words) < SHINGLE + MIN_SHINGLES:
        return []
    hashes = {zlib.crc32(" ".join(words[i:i + SHINGLE]).encode("utf-8")) for i in range(len(words) - SHINGLE + 1)}
    return heapq.nsmallest(SKETCH, hashes)


def jaccard_estimate(a: list[int], b: list[int]) -> float:
    union = heapq.nsmallest(SKETCH, set(a) | set(b))
    sa, sb = set(a), set(b)
    return sum(1 for h in union if h in sa and h in sb) / max(1, len(union))


def richness(record: dict) -> tuple:
    return (len(record["target"].get("alternatives") or []), len((record.get("contrasts") or {}).get("distractors") or []),
            sum(len(s.get("exact_refs") or []) for s in record["sources"]), len(source_text(record)))


def analyse(records: list[dict]) -> dict:
    """Return {'drop': [...ids], 'links': [[id, id, reason], ...], 'report': {...}}."""
    by_example: dict[str, list[dict]] = defaultdict(list)
    by_question: dict[str, list[str]] = defaultdict(list)
    sketches: dict[str, list[int]] = {}
    source_hash: dict[str, list[str]] = defaultdict(list)
    background = background_sources(records)
    for r in records:
        src = source_text(r)
        example = text_hash(src + "\n\u0000" + consumer_text(r) + "\n\u0000" + target_text(r))
        by_example[example].append(r)
        by_question[text_hash(consumer_text(r) + "\n\u0000" + target_text(r))].append(r["id"])
        own = source_text(r, background)
        if not own.strip():
            continue  # only background material: no document identity to compare
        source_hash[text_hash(own)].append(r["id"])
        sk = sketch(own)
        if sk:
            sketches[r["id"]] = sk
    drop, links = [], []
    for copies in by_example.values():
        if len(copies) > 1:
            keep = max(copies, key=richness)
            for c in copies:
                if c is not keep:
                    drop.append(c["id"])
                    links.append([keep["id"], c["id"], "exact-example"])
    for ids in by_question.values():
        for other in ids[1:]:
            links.append([ids[0], other, "same-question"])
    for ids in source_hash.values():
        for other in ids[1:]:
            links.append([ids[0], other, "same-source"])
    # Near duplicates among distinct source texts only.
    representative = {ids[0]: ids for ids in source_hash.values()}
    index: dict[int, list[str]] = defaultdict(list)
    near = 0
    seen_pairs = set()
    for rid in representative:
        sk = sketches.get(rid)
        if not sk:
            continue
        candidates = set()
        for h in sk[:BAND]:
            candidates.update(index[h])
            index[h].append(rid)
        for other in candidates:
            pair = (other, rid)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if jaccard_estimate(sk, sketches[other]) >= NEAR_THRESHOLD:
                links.append([other, rid, "near-duplicate-source"])
                near += 1
    report = {
        "records": len(records),
        "exact_example_duplicates_dropped": len(drop),
        "question_duplicate_links": sum(len(v) - 1 for v in by_question.values() if len(v) > 1),
        "same_source_links": sum(len(v) - 1 for v in source_hash.values() if len(v) > 1),
        "near_duplicate_source_links": near,
        "sketched_sources": len(sketches),
        "background_source_texts": len(background),
    }
    return {"drop": drop, "links": links, "report": report}


def write(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dedup.report.json").write_text(json.dumps(result["report"], indent=2) + "\n")
    with open(out_dir / "dedup.links.jsonl", "w", encoding="utf-8") as stream:
        for a, b, reason in result["links"]:
            stream.write(json.dumps({"a": a, "b": b, "reason": reason}) + "\n")
    (out_dir / "dedup.drop.json").write_text(json.dumps(sorted(result["drop"]), indent=1) + "\n")
