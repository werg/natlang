#!/usr/bin/env python3
"""Verify program IR in the current harness and emit template-neutral turns."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import run_program
from audit_trajectory_admission import audit_turns
from natlang.native import _strip_private
from program_ir import read_jsonl, lower
from natlang.trace import TraceReader, TraceRecorder
from natlang.scenario import ScenarioContract, admit
from natlang.gen.programs import BLOCKED

SOURCE_FILES = ("scripts/materialize_ir.py", "scripts/program_ir.py", "scripts/generate.py",
                "scripts/generate_external.py", "scripts/materialize_ir_shards.py")


def harness_hashes():
    repo = Path(__file__).resolve().parent.parent
    paths = [repo / name for name in SOURCE_FILES]
    paths.extend((repo / "natlang").rglob("*.py"))
    paths.extend((repo / "natlang" / "prompts").rglob("*.md"))
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths)) if path.exists()}


def materialize_record(record, recovery_rate=0, trace_dir=None):
    program = lower(record)
    ir_digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    recorder = None
    trace_file = None
    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / (hashlib.sha256(record["id"].encode()).hexdigest()[:24] + ".jsonl")
        recorder = TraceRecorder({"run_id": record["id"], "source_sha256": ir_digest,
                                  "semantic_version": record["version"], "tool_schema": "tools-v2",
                                  "engine_bindings": ["quickjs-isolated"],
                                  "capture": "reference-reduction", "seed_policy": "reference-agent"}, trace_file)
    try:
        samples, episodes = run_program(program, recovery_seed=record["id"],
                                        recovery_rate=recovery_rate, trace_sink=recorder)
    finally:
        if recorder is not None:
            recorder.close()
    if recorder is not None:
        reader = TraceReader(recorder.events)
        replay = reader.replay_observations()
        expected_kind = "quiesced" if program.outcome in ("blocked", "error") else "done"
        if program.expected == BLOCKED:
            expected_kind = "quiesced"
        expected_value = replay["final"] if callable(program.expected) or expected_kind != "done" else program.expected
        semantic_contract = record["semantics"].get("contract") if record["kind"] == "lambda_scenario" else None
        required = tuple({"name": item["tool"], "arguments": item["arguments"]}
                         for item in (semantic_contract or {}).get("required_actions", []))
        constraints = tuple((semantic_contract or {}).get("constrained_calls", []))
        effects = (tuple(("out.emit", [payload]) for payload in program.expected_effects)
                   if program.expected_effects is not None else None)
        admission = admit(reader, ScenarioContract(expected_kind, expected_value,
                                                   effects=effects, required_actions=required,
                                                   constrained_calls=constraints))
        admission["trace_sha256"] = hashlib.sha256(trace_file.read_bytes()).hexdigest()
    else:
        admission = None
    if record["kind"] == "lambda_scenario":
        audit_turns(record, samples)
    provenance = {key: record[key] for key in ("source", "split", "source_ids",
                   "source_groups", "source_revisions", "license", "gold_sources")}
    lines = []
    for turn, sample in enumerate(samples):
        lines.append({"id": f"{record['id']}:{turn}", "program_id": record["id"],
                      "family": record["kind"], "ir_version": record["version"],
                      "provisional_gold": bool(record["semantics"].get("contains_templates")),
                      "ir_digest": ir_digest,
                      **({"trace": {"file": str(trace_file), **admission}} if admission else {}),
                      **provenance, **sample, "tools": _strip_private(sample["tools"])})
    return lines, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-programs", type=int)
    parser.add_argument("--recovery-rate", type=float, default=0)
    parser.add_argument("--trace-dir", type=Path, help="write and admit whole-run reduction traces")
    args = parser.parse_args()
    if args.max_programs is not None and args.max_programs < 1:
        parser.error("max-programs must be positive")
    source_hashes = harness_hashes()
    ir_hash = hashlib.sha256(args.src.read_bytes()).hexdigest()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    opener = gzip.open if args.dst.suffix == ".gz" else open
    with opener(args.dst, "wt", encoding="utf-8") as out:
        for number, record in enumerate(read_jsonl(args.src)):
            if args.max_programs is not None and number >= args.max_programs:
                break
            if record["id"] in seen:
                raise ValueError(f"duplicate IR id: {record['id']}")
            seen.add(record["id"])
            lines, episodes = materialize_record(record, args.recovery_rate, args.trace_dir)
            for line in lines:
                out.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
                counts["turns"] += 1
                counts["provisional_turns" if line["provisional_gold"] else "eligible_turns"] += 1
                counts["skill:" + line["skill"]] += 1
                if args.trace_dir:
                    counts["trace_linked_turns"] += 1
            counts["programs"] += 1
            counts["provisional_programs" if record["semantics"].get("contains_templates") else "eligible_programs"] += 1
            counts["episodes"] += episodes
            if counts["programs"] % 1000 == 0:
                print(f"verified {counts['programs']} programs, {counts['turns']} turns", flush=True)
    manifest = {"materializer": "natlang.materializer/1", "ir_file": str(args.src),
                "ir_sha256": ir_hash, "harness_sha256": source_hashes,
                "recovery_rate": args.recovery_rate, "counts": dict(counts)}
    if args.trace_dir:
        manifest["trace_dir"] = str(args.trace_dir)
    args.dst.with_suffix(args.dst.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.dst), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
