#!/usr/bin/env python3
"""Render linked proposal-review IR from fresh base-program execution histories."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import run_program
from generate_agent_support import CHALLENGES
from materialize_ir import harness_hashes
from natlang.gen.policy import native_text
from natlang.tool_agent import review_messages, review_tools
from program_ir import digest, lower, read_jsonl
from natlang.corpus import file_digest


def render_review(record, base, samples):
    sem = record["semantics"]
    if sem["base_program_digest"] != digest(base):
        raise ValueError(f"base program changed: {record['id']}")
    sample = samples[sem["turn_index"]]
    proposal = sem["proposal"]
    if sem["proposal_justified"]:
        actual = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                  for c in sample["target"].get("tool_calls", [])]
        if proposal != [[name, args] for name, args in actual]:
            raise ValueError(f"approved proposal differs from fresh reference: {record['id']}")
    verdict = {"reason": sem["reason"], "decision": sem["verdict"]}
    messages = review_messages(sample["messages"], proposal, sem["check_call_index"],
                               sem["prompt_variant"])
    messages[-1]["content"] += "\n\n" + CHALLENGES[sem["challenge_index"]]
    target = {"role": "assistant", "content": "", "tool_calls": [{
        "type": "function", "function": {"name": "review_write",
        "arguments": json.dumps(verdict, ensure_ascii=False)}}]}
    return {"id": record["id"], "program_id": record["id"],
            "base_program_id": base["id"], "ir_version": record["version"],
            "ir_digest": digest(record), "source": record["source"],
            "split": record["split"], "source_ids": record["source_ids"],
            "source_groups": record["source_groups"], "license": record["license"],
            "contrast_group": sem["contrast_group"], "lesson_ids": sem["lesson_ids"],
            "proposal_justified": sem["proposal_justified"],
            "task_feasible": sem["task_feasible"], "confidence": None,
            "messages": messages, "tools": review_tools(), "proposal": proposal,
            "target": target, "native_target": native_text([("review_write", verdict)]),
            "skill": "review_write", "kind": "review"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reviews_ir", type=Path)
    ap.add_argument("base_ir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-groups", type=int)
    args = ap.parse_args()
    bases = {r["id"]: r for r in read_jsonl(args.base_ir)}
    reviews = defaultdict(list)
    groups = set()
    for record in read_jsonl(args.reviews_ir):
        base_id = record["semantics"]["base_program_id"]
        group = record["source_groups"][0]
        if args.max_groups is not None and group not in groups:
            if len(groups) >= args.max_groups:
                continue
            groups.add(group)
        reviews[base_id].append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged = args.out.with_suffix(args.out.suffix + ".building")
    opener = gzip.open if args.out.suffix == ".gz" else open
    counts = Counter()
    with opener(staged, "wt", encoding="utf-8") as stream:
        for base_id, records in reviews.items():
            base = bases[base_id]
            samples, _ = run_program(lower(base))
            for record in records:
                row = render_review(record, base, samples)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["target"]["tool_calls"][0]["function"]["name"]] += 1
                counts["reviews"] += 1
    staged.replace(args.out)
    manifest = {"materializer": "natlang.review-materializer/1",
                "review_ir_sha256": file_digest(args.reviews_ir),
                "base_ir_sha256": file_digest(args.base_ir), "harness_sha256": harness_hashes(),
                "review_renderer_sources_sha256": {
                    name: file_digest(Path(__file__).resolve().parent.parent / name)
                    for name in ("scripts/materialize_review_ir.py", "scripts/generate_agent_support.py")},
                "counts": dict(counts), "output_sha256": file_digest(args.out)}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
