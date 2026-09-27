"""Merge the SFT turns of one conversation into one training sequence with a completion span per turn.

    python scripts/merge_sft_chains.py SFT.jsonl MERGED.jsonl

Each exported turn is its whole conversation so far (prompt) and the model's next message (completion), so a call of n
turns is trained on its opening n times. When a turn's prompt begins with an earlier turn's prompt and completion
(a later turn of the same call), the two share one sequence: the text between them carries no loss, and each
completion keeps its own. The loss targets are the same tokens; the sequences to process are about a third as long.
A turn whose prompt extends none (a call's first, a child call, or a history compacted since) starts a new sequence.

Rows keep id (the first turn's), program_id, source_groups and family, list their turns, and carry `segments`:
[text, trained] pairs, alternating untrained context and trained completion. train_lora.py trains them.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge(rows):
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[row.get("teacher_trajectory_id") or row["id"]].append(row)
    merged = []
    for turns in by_trajectory.values():
        chains = []  # [text so far, row, segments, turn ids]
        for row in sorted(turns, key=lambda r: (len(r["prompt"]), r["id"])):
            prompt, completion = row["prompt"], row["completion"]
            found = max((c for c in chains if prompt.startswith(c[0])), key=lambda c: len(c[0]), default=None)
            if found is None:
                chains.append([prompt + completion, row, [[prompt, False], [completion, True]], [row["id"]]])
                continue
            found[2] += [[prompt[len(found[0]):], False], [completion, True]]
            found[0] = prompt + completion
            found[3].append(row["id"])
        for text, first, segments, ids in chains:
            merged.append({"id": first["id"], "program_id": first.get("program_id"),
                           "source_groups": first.get("source_groups", []), "family": first.get("family", "unknown"),
                           **({"split": first["split"]} if "split" in first else {}),
                           "teacher_trajectory_id": first.get("teacher_trajectory_id"), "turns": ids,
                           "segments": [s for s in segments if s[0] or s[1]]})
    return merged


def main(source: Path, out: Path):
    rows = [json.loads(line) for line in source.open() if line.strip()]
    merged = merge(rows)
    with out.open("w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")
    before = sum(len(r["prompt"]) + len(r["completion"]) for r in rows)
    after = sum(len(t) for r in merged for t, _ in r["segments"])
    print(f"{len(rows)} turns -> {len(merged)} sequences; {before / 1e6:.1f}M -> {after / 1e6:.1f}M characters -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    main(a.source, a.out)
