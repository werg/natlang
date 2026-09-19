#!/usr/bin/env python3
"""Apply checked generative-leaf references to frozen synthetic IR and replay gold."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import run_program
from materialize_ir import harness_hashes
from natlang.corpus import file_digest
from natlang.gen.codebases import ref_key
from natlang.values import dump
from program_ir import VERSION, digest, lower, read_jsonl, validate


def load_references(path: Path):
    values = {}
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = ref_key(row["function"], row["args"])
            if row["key"] != key:
                raise ValueError(f"{path}:{number}: reference key does not match function and arguments")
            if not isinstance(row.get("value"), str):
                raise ValueError(f"{path}:{number}: generative leaf reference must be text")
            if key in values and values[key] != row["value"]:
                raise ValueError(f"{path}:{number}: conflicting reference for {key}")
            values[key] = row["value"]
    return values


def replace_oracles(record, references):
    if not record["semantics"].get("contains_templates"):
        return record, 0
    updated = copy.deepcopy(record)
    replaced = 0
    for function, oracle in updated["semantics"].get("leaf_oracles", {}).items():
        for case in oracle.get("cases", []):
            if not case.get("template"):
                continue
            key = ref_key(function, case["input"])
            if key in references:
                case["output"] = references[key]
                case["template"] = False
                case["reference_key"] = key
                replaced += 1
    updated["semantics"]["contains_templates"] = any(
        case.get("template") for oracle in updated["semantics"].get("leaf_oracles", {}).values()
        for case in oracle.get("cases", []))
    return updated, replaced


def replay_gold(record):
    """Run the frozen source graph with its new leaf answers to capture value/effects."""
    program = lower(record)
    values = []
    program.expected = lambda value: values.append(value) or True
    observed = {}
    for name, original in list(program.capabilities.items()):
        spec = record["semantics"]["effects"][name]
        if not isinstance(spec, list) and spec.get("kind") != "record_args":
            raise ValueError(f"cannot refresh effect contract {name}: {spec}")
        observed[name] = []
        full_args = isinstance(spec, dict)
        def recording(args, original=original, bucket=observed[name], full_args=full_args):
            result = original(args)
            bucket.append(dump(args if full_args else args[0]))
            return result
        program.capabilities[name] = recording
    run_program(program)
    if len(values) != 1:
        raise ValueError(f"expected one replayed gold value, got {len(values)}")
    return values[0], observed


def refresh(record, references, reference_sha256):
    updated, replaced = replace_oracles(record, references)
    if not replaced:
        return record, 0
    expected, effects = replay_gold(updated)
    updated["semantics"]["expected"] = expected
    for name, values in effects.items():
        spec = updated["semantics"]["effects"][name]
        if isinstance(spec, list):
            updated["semantics"]["effects"][name] = values
        else:
            spec["expected"] = values
    updated["gold_sources"] = sorted(set(updated.get("gold_sources", [])) | {"checked-teacher-reference"})
    updated["reference_refresh"] = {"base_ir_digest": digest(record),
                                    "reference_bank_sha256": reference_sha256,
                                    "replaced_leaf_cases": replaced}
    validate(updated)
    run_program(lower(updated))  # verify new gold and ordered effects independently
    return updated, replaced


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="frozen synthetic IR JSONL")
    ap.add_argument("dst", type=Path, help="new IR JSONL; source is never overwritten")
    ap.add_argument("--references", type=Path, default=Path("data/leaf_references.jsonl"))
    ap.add_argument("--max-programs", type=int)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    if args.src.resolve() == args.dst.resolve():
        ap.error("source and destination must differ")
    if args.dst.exists() or args.dst.with_suffix(args.dst.suffix + ".manifest.json").exists():
        ap.error("destination already exists; choose a versioned output path")
    if args.max_programs is not None and args.max_programs < 1:
        ap.error("max-programs must be positive")
    reference_hash = file_digest(args.references)
    references = load_references(args.references)
    if file_digest(args.references) != reference_hash:
        raise ValueError("reference bank changed while it was being read; retry")
    source_hash = file_digest(args.src)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    staged = args.dst.with_suffix(args.dst.suffix + ".building")
    counts = Counter()
    output_hash = hashlib.sha256()
    with staged.open("w") as stream:
        for record in read_jsonl(args.src):
            if args.max_programs is not None and counts["programs"] >= args.max_programs:
                break
            if record.get("source") != "natlang-synthetic":
                raise ValueError(f"unexpected source: {record['id']}")
            updated, replaced = refresh(record, references, reference_hash)
            counts["programs"] += 1
            counts["refreshed_programs"] += bool(replaced)
            counts["replaced_leaf_cases"] += replaced
            counts["provisional_programs"] += bool(updated["semantics"].get("contains_templates"))
            line = json.dumps(updated, ensure_ascii=False) + "\n"
            stream.write(line)
            output_hash.update(line.encode())
            if counts["programs"] % 1000 == 0:
                print(f"refreshed {counts['programs']} programs", flush=True)
    if args.require_complete and counts["provisional_programs"]:
        raise ValueError(f"{counts['provisional_programs']} programs still have template gold; staged output retained")
    staged.replace(args.dst)
    counts["eligible_programs"] = counts["programs"] - counts["provisional_programs"]
    manifest = {"version": VERSION, "base_ir_sha256": source_hash,
                "reference_bank_sha256": reference_hash, "counts": dict(counts),
                "ir_sha256": output_hash.hexdigest(),
                "replay_harness_sha256": harness_hashes(),
                "refresher_sha256": file_digest(Path(__file__))}
    args.dst.with_suffix(args.dst.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
