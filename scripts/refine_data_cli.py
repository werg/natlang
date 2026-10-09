#!/usr/bin/env python3
"""Refinement-type training data (plans/REFINEMENT_DATA.md). No command loads a model.

  harvest             predicates of the applications' DECOMPOSITION.md tables -> seeds.jsonl
  exact               refine-judge source (c): exact-label corpus directory (rows, predicates, provenance manifest)
  mine                refine-judge source (a): unlabeled pairs from call stores and trace files
  exemplify-requests / ingest-exemplify / near-miss-requests / verify-requests / accept / label-requests / assemble
                      refine-judge source (b) and the labelling of (a): requests for the teacher stages, and the assembly
  author              refine-author corpus directory
  calibration         queue | review | status | export | report   (refine-calibration)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_data import author, calibration, exact, judge, miner, predicates  # noqa: E402
from refine_data.common import (HELDOUT_MODULUS, HELDOUT_REMAINDERS, REPO, SPLIT_RULE_VERSION, read_jsonl,  # noqa: E402
                                utc_now_iso, write_jsonl)

SPLIT_RULE = (f"{SPLIT_RULE_VERSION}: a predicate text, whitespace-normalised, goes to bucket int(sha256('{SPLIT_RULE_VERSION}|' + text)[:8], 16) % "
              f"{HELDOUT_MODULUS}; buckets below {HELDOUT_REMAINDERS} are held out (split heldout-predicate, role heldout). Every row of a predicate "
              "has the same split, whatever its source or value. Families listed under heldout_families are held out whole "
              "(split heldout-family, role heldout). All other rows are split train, role train.")


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def provenance(corpus_id: str, family: str, extra: dict) -> dict:
    return {"schema": "natlang.refine-corpus-provenance/1", "id": corpus_id, "family": family, "generated_at": utc_now_iso(),
            "generator_commit": git_head(), "admission": "held-pending-review", "training_admission": False,
            "model_calls": 0, "notes": "Admission is a separate decision (plans/REFINEMENT_DATA.md, admission criteria).", **extra}


def dump(path: Path, value) -> None:
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def cmd_harvest(args):
    found, skipped = predicates.harvest()
    write_jsonl(args.out, predicates.unique_predicates(found))
    print(json.dumps({"predicates": len(found), "skipped": skipped}))


def cmd_exact(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True)
    rows, report = exact.generate(per_predicate=args.per_predicate)
    n = write_jsonl(out / "judge-rows.jsonl", rows)
    specs = exact.all_specs()
    write_jsonl(out / "predicates.jsonl", [{"predicate_id": exact.predicate_id(s.predicate), "predicate": s.predicate, "family": s.family, "real": s.real,
                                            "split": exact.split_of(s.predicate, s.family, exact.HELDOUT_FAMILIES)} for s in specs])
    splits = {}
    for row in rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    dump(out / "manifest.json", provenance(args.id, "refine-judge", {
        "source": "exact", "format": "natlang.decision-prompt/1 rows (judge-rows.jsonl); one predicate per line in predicates.jsonl",
        "label_source": "crisp checkers (scripts/refine_data/exact.py); the label of a row is its checker's verdict",
        "split_rule": SPLIT_RULE, "heldout_families": sorted(exact.HELDOUT_FAMILIES), "rows": n, "rows_by_split": splits,
        "predicates": report["predicates"], "per_predicate": args.per_predicate, "seed": "refine-judge-exact/1",
        "dropped_intent_mismatch": report["dropped_intent_mismatch"],
        "judge_prompt": "mirrors decisionJudge (ts-host/src/native/refinement.ts); parity-tested"}))
    print(json.dumps({"rows": n, "splits": splits, **report}))


def cmd_mine(args):
    readers = [miner.StoreReader(Path(p)) for p in args.store]
    try:
        sources = [item for reader in readers for item in miner.events_of_store(reader)]
        sources += [item for p in args.trace for item in miner.events_of_trace_file(Path(p))]
        pairs, stats = miner.mine(sources, readers)
    finally:
        for reader in readers:
            reader.close()
    write_jsonl(args.out, pairs)
    print(json.dumps(stats))


def _candidates():
    found, _ = predicates.harvest()
    return predicates.unique_predicates(found)


def cmd_exemplify_requests(args):
    print(json.dumps({"requests": write_jsonl(args.out, judge.exemplify_requests(_candidates(), args.count))}))


def cmd_ingest_exemplify(args):
    requests = read_jsonl(args.requests)
    rows = judge.satisfying_from_exemplify(requests, read_jsonl(args.results))
    print(json.dumps({"satisfying": write_jsonl(args.out, rows)}))


def cmd_near_miss_requests(args):
    print(json.dumps({"requests": write_jsonl(args.out, judge.near_miss_requests(read_jsonl(args.satisfying)))}))


def cmd_verify_requests(args):
    rows = judge.verify_requests(read_jsonl(args.satisfying), read_jsonl(args.near_miss_results))
    print(json.dumps({"requests": write_jsonl(args.out, rows)}))


def cmd_accept(args):
    pairs, stats = judge.accept_near_misses(read_jsonl(args.satisfying), read_jsonl(args.near_miss_results), read_jsonl(args.verify_results))
    write_jsonl(args.out, pairs)
    print(json.dumps(stats))


def cmd_label_requests(args):
    accepted = read_jsonl(args.accepted) if args.accepted else []
    mined = read_jsonl(args.mined) if args.mined else []
    print(json.dumps({"pairs": write_jsonl(args.out, judge.label_requests(accepted, mined))}))


def cmd_assemble(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True)
    accepted = read_jsonl(args.accepted) if args.accepted else []
    mined = read_jsonl(args.mined) if args.mined else []
    labels = [row for path in args.labels for row in read_jsonl(path)]
    rows, disagreements, report = judge.assemble(accepted, mined, labels, args.teacher)
    write_jsonl(out / "judge-rows.jsonl", rows)
    write_jsonl(out / "disagreements.jsonl", disagreements)
    dump(out / "manifest.json", provenance(args.id, "refine-judge", {
        "source": "near-miss+mined", "teacher": args.teacher, "split_rule": SPLIT_RULE, "report": report,
        "label_source": "near-miss: construction + independent verification pass (gold), teacher judge P(true) in teachers; "
                        "mined: teacher judge P(true) (gold and teachers)"}))
    print(json.dumps(report))


def cmd_author(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True)
    examples, omissions, report = author.generate()
    write_jsonl(out / "examples.jsonl", examples)
    write_jsonl(out / "omissions.jsonl", omissions)
    dump(out / "manifest.json", provenance(args.id, "refine-author", {
        "source": "applications/refine-data/authoring.json over the working tree and git history",
        "format": "natlang.refine-author/1: before/after function and types.ts, unified diffs, refined types, user/assistant messages",
        "split_rule": "No train/held-out split: the corpus is small and static; hold out by app when training (the app field), "
                      "never by example, because one function's examples share its types.ts",
        "report": report, "origins": sorted({e["origin"].get("rev", "tree") for e in examples})}))
    print(json.dumps({k: v for k, v in report.items() if k != "uncovered_predicate_ids"}))


def cmd_calibration(args):
    if args.action == "queue":
        queue = calibration.build_queue(read_jsonl(args.rows), args.per_predicate)
        print(json.dumps({"items": write_jsonl(args.out, queue)}))
    elif args.action == "review":
        print(json.dumps(calibration.review(read_jsonl(args.queue), Path(args.verdicts), args.reviewer, limit=args.limit)))
    elif args.action == "status":
        verdicts = read_jsonl(args.verdicts) if Path(args.verdicts).exists() else []
        print(json.dumps(calibration.status(read_jsonl(args.queue), verdicts), indent=2))
    elif args.action == "export":
        print(json.dumps({"rows": write_jsonl(args.out, calibration.to_rows(read_jsonl(args.verdicts)))}))
    elif args.action == "report":
        predictions = {row["id"]: row["p_true"] for row in read_jsonl(args.predictions) if "p_true" in row}
        print(json.dumps(calibration.calibration_report(read_jsonl(args.verdicts), predictions), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, *options):
        p = sub.add_parser(name)
        for flag, kwargs in options:
            p.add_argument(flag, **kwargs)
        p.set_defaults(fn=fn)
        return p

    out = ("--out", {"required": True})
    add("harvest", cmd_harvest, out)
    add("exact", cmd_exact, ("--out-dir", {"required": True}), ("--id", {"required": True}), ("--per-predicate", {"type": int, "default": 16}))
    add("mine", cmd_mine, out, ("--store", {"action": "append", "default": []}), ("--trace", {"action": "append", "default": []}))
    add("exemplify-requests", cmd_exemplify_requests, out, ("--count", {"type": int, "default": 6}))
    add("ingest-exemplify", cmd_ingest_exemplify, out, ("--requests", {"required": True}), ("--results", {"required": True}))
    add("near-miss-requests", cmd_near_miss_requests, out, ("--satisfying", {"required": True}))
    add("verify-requests", cmd_verify_requests, out, ("--satisfying", {"required": True}), ("--near-miss-results", {"required": True}))
    add("accept", cmd_accept, out, ("--satisfying", {"required": True}), ("--near-miss-results", {"required": True}), ("--verify-results", {"required": True}))
    add("label-requests", cmd_label_requests, out, ("--accepted", {}), ("--mined", {}))
    add("assemble", cmd_assemble, ("--out-dir", {"required": True}), ("--id", {"required": True}), ("--teacher", {"required": True}),
        ("--accepted", {}), ("--mined", {}), ("--labels", {"action": "append", "required": True}))
    add("author", cmd_author, ("--out-dir", {"required": True}), ("--id", {"required": True}))
    p = add("calibration", cmd_calibration, ("--action", {"required": True, "choices": ["queue", "review", "status", "export", "report"]}),
            ("--rows", {}), ("--out", {}), ("--per-predicate", {"type": int, "default": 4}), ("--queue", {}), ("--verdicts", {}),
            ("--reviewer", {"default": "reviewer"}), ("--limit", {"type": int}), ("--predictions", {}))
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
