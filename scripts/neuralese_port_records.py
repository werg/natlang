#!/usr/bin/env python3
"""Build, validate, deduplicate and split-close Neuralese port records (plans/neuralese/S1_DATA.md).

Subcommands:
  ledger            write the dataset ledger (training/neuralese_data_ledger.json)
  convert-bgkit     bgkit task stores → port records
  convert-schnitzel Schnitzeljagd tasks-* corpora → port records
  convert-swe       SWE-rebench OpenHands trajectories → continuation records
  validate          check records against natlang.port-record/1, markup and leakage rules
  dedup             exact and near-duplicate analysis across record files
  protected         build the protected held-out set from bgkit benchmarks
  close             split-group closure with dedup links and protected hits; writes closed copies
  audit-splits      stream-check split-group conflicts and protected-set placement
Converters reading parquet need pyarrow (for example /home/werg/bgkit/.venv/bin/python).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuralese_data import agents, bgkit, dedup, finalize, inventory, schnitzel, schnitzel_turns, splits, swe_compaction, trajectory  # noqa: E402
from neuralese_data.common import DEFAULT_OUTPUT_ROOT, Reject, Sink  # noqa: E402
from neuralese_data.records import (SCHEMA_PATH, VERSION, canonical_json, leakage, require_schema_validator, validate,
                                    validate_with_schema)  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _advise_stream(stream):
    if hasattr(os, 'posix_fadvise'):
        for hint in ('POSIX_FADV_SEQUENTIAL', 'POSIX_FADV_NOREUSE'):
            if hasattr(os, hint):
                try:
                    os.posix_fadvise(stream.fileno(), 0, 0, getattr(os, hint))
                except OSError:
                    pass  # Optional cache advice; reads remain identical.


def _records(paths):
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            _advise_stream(stream)
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _record_files(inputs: list[str]) -> list[Path]:
    out = []
    for raw in inputs:
        p = Path(raw)
        out.extend(sorted(p.glob("*.port-records.jsonl")) if p.is_dir() else [p])
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        _advise_stream(stream)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_summary(out_dir: Path, name: str, summaries: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.summary.json"
    merged = {s["name"]: s for s in (json.loads(path.read_text()) if path.exists() else [])}
    merged.update({s["name"]: s for s in summaries})  # a partial rerun replaces only its own entries
    path.write_text(json.dumps(sorted(merged.values(), key=lambda s: s["name"]), indent=2, ensure_ascii=False) + "\n")
    for s in summaries:
        print(f"{s['name']}: accepted {s['accepted']} {s['by_split']} rejected {s['rejected']}")


def cmd_ledger(args):
    ledger = inventory.build()
    args.output.write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(ledger["counts"]), "missing reasons:", len(ledger["missing_reason"]))


def cmd_convert_bgkit(args):
    task_repos = bgkit.load_task_repos()
    heldout = bgkit.load_heldout_repos()
    summaries = []
    for store in args.stores or bgkit.ALL_STORES:
        convert = bgkit.converter_for(store, task_repos, heldout)
        sink = Sink(args.out, f"bgkit.{store}")
        for i, row in _bgkit_rows(store, args.limit, args.seed):
            try:
                sink.accept(convert(i, row))
            except Reject as exc:
                sink.reject(store, i, str(exc))
        summaries.append(sink.close())
    _write_summary(args.out, "bgkit", summaries)


def _bgkit_rows(store: str, limit: int | None, seed: int):
    from neuralese_data.common import parquet_rows

    return parquet_rows(bgkit.TASKS_DIR / f"{store}.parquet", limit=limit, seed=seed)


def cmd_convert_schnitzel(args):
    summaries = []
    for corpus in args.corpora or schnitzel.CORPORA:
        sink = Sink(args.out, f"schnitzeljagd.{schnitzel.corpus_short(corpus)}")
        for split, i, episode, texts in schnitzel.iter_corpus(corpus, args.limit):
            try:
                sink.accept(schnitzel.convert_episode(corpus, split, i, episode, texts))
            except Reject as exc:
                sink.reject(corpus, f"{split}:{i}", str(exc))
        summaries.append(sink.close())
    _write_summary(args.out, "schnitzeljagd", summaries)


def cmd_convert_swe(args):
    bench = swe_compaction.load_bench_instances()
    sink = Sink(args.out, "upstream.swe-rebench-openhands")
    for i, row in swe_compaction.iter_rows(args.limit, args.seed):
        picks = swe_compaction.windows_for(row["trajectory"], args.windows, row["trajectory_id"])
        if not picks:
            sink.reject(swe_compaction.UPSTREAM, i, "too-short-trajectory")
            continue
        for recent_start, target in picks:
            try:
                sink.accept(swe_compaction.convert_window(i, row, recent_start, target, bench))
            except Reject as exc:
                sink.reject(swe_compaction.UPSTREAM, f"{i}:{target}", str(exc))
    _write_summary(args.out, "swe", [sink.close()])


def cmd_convert_agents(args):
    summaries = []
    for name in args.corpora or list(agents.CORPORA):
        adapter, windows = agents.CORPORA[name]
        sink = Sink(args.out, f"upstream.{name}")
        try:
            for row, traj in adapter(args.limit):
                trajectory.records_for(traj, args.windows or windows, sink, name, row)
        except Reject as exc:  # an adapter-level rejection ends the corpus with a recorded reason
            sink.reject(name, "adapter", str(exc))
        summaries.append(sink.close())
    _write_summary(args.out, "agents", summaries)


def cmd_convert_turns(args):
    summaries = []
    for corpus in args.corpora or schnitzel_turns.CORPORA:
        sink = Sink(args.out, f"schnitzeljagd.{schnitzel.corpus_short(corpus)}")
        for split, i, episode in schnitzel_turns.iter_corpus(corpus, args.limit):
            try:
                traj = schnitzel_turns.trajectory(corpus, split, i, episode)
            except Reject as exc:
                sink.reject(corpus, f"{split}:{i}", str(exc))
                continue
            trajectory.records_for(traj, args.windows, sink, corpus, f"{split}:{i}")
        summaries.append(sink.close())
    _write_summary(args.out, "schnitzeljagd-turns", summaries)


def cmd_finalize(args):
    protected = json.loads(args.protected.read_text()) if args.protected else None
    report = finalize.build([Path(p) for p in args.inputs], args.out, protected)
    print(json.dumps({k: v for k, v in report.items() if k not in ("families", "licences", "inputs")}, indent=2))
    return 1 if report["closure_violations"] or report["invalid"] else 0


def cmd_validate(args):
    schema_sha256 = None
    if args.schema:
        try:
            _validator, schema_sha256 = require_schema_validator()
        except ImportError:
            print("validate --schema requires the 'jsonschema' package; install it or omit --schema", file=sys.stderr)
            return 2
    check = validate_with_schema if args.schema else validate
    totals, failures, ids = Counter(), Counter(), Counter()
    examples = {}
    for path in _record_files(args.inputs):
        for record in _records([path]):
            totals[path.name] += 1
            ids[record.get("id")] += 1
            errors = check(record)
            if not errors:
                errors += leakage(record)
            for e in errors:
                key = e.split(":")[0]
                failures[key] += 1
                examples.setdefault(key, f"{record.get('id')}: {e}")
    dup_ids = {k: v for k, v in ids.items() if v > 1}
    report = {"files": dict(totals), "records": sum(totals.values()), "errors": dict(failures),
              "error_examples": examples, "duplicate_ids": len(dup_ids),
              "schema_validation": ({"applied": True, "path": str(SCHEMA_PATH), "sha256": schema_sha256}
                                    if args.schema else {"applied": False})}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 1 if failures or dup_ids else 0


def _protected_values(value, label: str) -> set[str]:
    if isinstance(value, dict):
        values = value.keys()
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"protected file {label} must be an object or array of strings")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"protected file {label} contains a non-string or empty value")
    return set(values)


def cmd_audit_splits(args):
    try:
        protected_bytes = args.protected.read_bytes()
        protected_raw = json.loads(protected_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"cannot read protected set {args.protected}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(protected_raw, dict) or "question_hashes" not in protected_raw or "ids" not in protected_raw:
        print("protected JSON must contain both 'question_hashes' and 'ids'", file=sys.stderr)
        return 2
    try:
        question_hashes = _protected_values(protected_raw["question_hashes"], "question_hashes")
        protected_ids = _protected_values(protected_raw["ids"], "ids")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in question_hashes):
        print("protected question hashes must be lowercase SHA-256 hex digests", file=sys.stderr)
        return 2
    if not question_hashes and not protected_ids:
        print("protected set is empty (both question_hashes and ids are empty)", file=sys.stderr)
        return 2
    protected = {"question_hashes": question_hashes, "ids": protected_ids}
    input_files = _record_files(args.inputs)
    if not input_files:
        print("no record input files matched", file=sys.stderr)
        return 2

    split_names = ("train", "validation", "test")
    split_bits = {name: 1 << i for i, name in enumerate(split_names)}
    group_split_bits: dict[str, int] = {}
    split_counts = Counter()
    protected_by_split = Counter()
    protected_by_kind = Counter()
    invalid_counts = Counter()
    examples: list[dict] = []
    input_receipts = []
    total_rows = 0

    def fail(kind: str, path: Path, line_number: int, record_id, detail: str) -> None:
        invalid_counts[kind] += 1
        if len(examples) < 10:
            examples.append({"kind": kind, "path": str(path), "record_ordinal": line_number,
                             "record_id": record_id, "detail": detail})

    for path in input_files:
        before = path.stat()
        file_rows = 0
        for line_number, record in enumerate(_records([path]), 1):
            file_rows += 1
            total_rows += 1
            if not isinstance(record, dict):
                fail("invalid_record", path, line_number, None, "record must be a JSON object")
                continue
            record_id = record.get("id")
            split = record.get("split")
            groups = record.get("split_groups")
            if record.get("version") != VERSION:
                fail("record_version", path, line_number, record_id,
                     f"expected {VERSION}, got {record.get('version')!r}")
            split_valid = isinstance(split, str) and split in split_bits
            if not split_valid:
                fail("invalid_split", path, line_number, record_id, f"unsupported split {split!r}")
            if (not isinstance(groups, list) or not groups or
                    any(not isinstance(group, str) or not group for group in groups)):
                fail("invalid_split_groups", path, line_number, record_id,
                     "split_groups must be a nonempty array of nonempty strings")
            elif split_valid:
                bit = split_bits[split]
                for group in groups:
                    group_split_bits[group] = group_split_bits.get(group, 0) | bit
            split_counts[split if split_valid else "invalid"] += 1
            try:
                hit = splits.protected_hit(record, protected)
            except (KeyError, TypeError, AttributeError) as exc:
                fail("protected_hit_error", path, line_number, record_id,
                     f"protected_hit could not inspect record: {type(exc).__name__}")
                continue
            if hit:
                protected_by_split[split if split_valid else "invalid"] += 1
                protected_by_kind[hit] += 1
                if split != "test":
                    fail("protected_hit_outside_test", path, line_number, record_id,
                         f"protected match kind {hit!r} is assigned to split {split!r}")
        digest = _sha256_file(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            fail("input_changed_during_audit", path, 0, None, "input metadata changed while scanning/hashing")
        input_receipts.append({"path": str(path.resolve()), "sha256": digest, "records": file_rows})

    if not total_rows:
        invalid_counts["empty_dataset"] += 1

    sampled_group_conflicts = 0
    conflict_count = 0
    group_digest_builder = hashlib.sha256()
    for group in sorted(group_split_bits):
        bitset = group_split_bits[group]
        memberships = [name for name in split_names if bitset & split_bits[name]]
        group_digest_builder.update(canonical_json([group, memberships]).encode("utf-8"))
        group_digest_builder.update(b"\n")
        if len(memberships) > 1:
            conflict_count += 1
            if len(examples) < 10:
                examples.append({"kind": "split_group_conflict", "split_group": group,
                                 "splits": memberships})
                sampled_group_conflicts += 1

    group_digest = group_digest_builder.hexdigest()
    protected_membership = {"question_hashes": sorted(question_hashes), "ids": sorted(protected_ids)}
    protected_digest = hashlib.sha256(canonical_json(protected_membership).encode("utf-8")).hexdigest()
    schema_sha256 = _sha256_file(SCHEMA_PATH)
    report = {
        "schema": "neuralese.audit-splits/1",
        "record_schema": {"version": VERSION, "path": str(SCHEMA_PATH.resolve()), "sha256": schema_sha256,
                          "full_json_schema_validation": False},
        "inputs": input_receipts,
        "records": total_rows,
        "records_by_split": dict(split_counts),
        "audit_errors": dict(invalid_counts),
        "split_groups": {"unique": len(group_split_bits), "cross_split_conflicts": conflict_count,
                         "membership_sha256": group_digest,
                         "conflict_examples_in_failure_sample": sampled_group_conflicts},
        "protected": {"path": str(args.protected.resolve()), "file_sha256": hashlib.sha256(protected_bytes).hexdigest(),
                      "question_hash_count": len(question_hashes), "id_count": len(protected_ids),
                      "membership_sha256": protected_digest, "hits_by_split": dict(protected_by_split),
                      "hits_by_match_kind": dict(protected_by_kind),
                      "non_test_hit_count": invalid_counts.get("protected_hit_outside_test", 0)},
        "failure_examples": examples,
        "examples_limit": 10,
        "limitations": ["Audits split-group cross-split collisions and protected-set placement only.",
                        "Does not establish background-source alias closure, provenance completeness, licensing quality, or full record-schema validity."],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "records": report["records"],
                      "split_group_conflicts": conflict_count,
                      "protected_non_test_hits": report["protected"]["non_test_hit_count"],
                      "errors": dict(invalid_counts)}, indent=2))
    return 1 if conflict_count or invalid_counts else 0


def cmd_dedup(args):
    records = list(_records(_record_files(args.inputs)))
    result = dedup.analyse(records)
    dedup.write(result, args.out)
    print(json.dumps(result["report"], indent=2))


def cmd_protected(args):
    protected = splits.build_protected(splits.default_protected_paths())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protected) + "\n")
    print(f"protected: {len(protected['question_hashes'])} question hashes, {len(protected['ids'])} ids")


def cmd_close(args):
    files = _record_files(args.inputs)
    records = list(_records(files))
    drop = set(json.loads(args.drop.read_text())) if args.drop else set()
    records = [r for r in records if r["id"] not in drop]
    links = []
    if args.links:
        links = [(x["a"], x["b"]) for x in _records([args.links])]
    protected = json.loads(args.protected.read_text()) if args.protected else None
    report = splits.close(records, links, protected)
    report["dropped_duplicates"] = len(drop)
    report["closure_violations"] = splits.check_closed(records)[:20]
    report["families"] = splits.family_counts(records)
    args.out.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list] = {}
    for r in records:
        by_file.setdefault(r["family"], []).append(r)
    for family, rows in by_file.items():
        with open(args.out / f"{family}.port-records.jsonl", "w", encoding="utf-8") as stream:
            for r in rows:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out / "closure.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "families"}, indent=2))
    return 1 if report["closure_violations"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ledger")
    p.add_argument("--output", type=Path, default=REPO / "training" / "neuralese_data_ledger.json")
    p.set_defaults(func=cmd_ledger)
    sample_root = DEFAULT_OUTPUT_ROOT / "port-records" / "samples"
    p = sub.add_parser("convert-bgkit")
    p.add_argument("--stores", nargs="*")
    p.add_argument("--limit", type=int, default=None, help="rows per store (sample); default all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=sample_root)
    p.set_defaults(func=cmd_convert_bgkit)
    p = sub.add_parser("convert-schnitzel")
    p.add_argument("--corpora", nargs="*")
    p.add_argument("--limit", type=int, default=None, help="train episodes per corpus; held-out splits get a fifth")
    p.add_argument("--out", type=Path, default=sample_root)
    p.set_defaults(func=cmd_convert_schnitzel)
    p = sub.add_parser("convert-swe")
    p.add_argument("--limit", type=int, default=None, help="trajectories")
    p.add_argument("--windows", type=int, default=2, help="continuation windows per trajectory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=sample_root)
    p.set_defaults(func=cmd_convert_swe)
    p = sub.add_parser("convert-agents")
    p.add_argument("--corpora", nargs="*", choices=list(agents.CORPORA))
    p.add_argument("--limit", type=int, default=None, help="trajectories per corpus; default all")
    p.add_argument("--windows", type=int, default=None, help="windows per trajectory; default per corpus")
    p.add_argument("--out", type=Path, default=sample_root)
    p.set_defaults(func=cmd_convert_agents)
    p = sub.add_parser("convert-turns")
    p.add_argument("--corpora", nargs="*")
    p.add_argument("--limit", type=int, default=None, help="train episodes per corpus; held-out splits get a fifth")
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--out", type=Path, default=sample_root)
    p.set_defaults(func=cmd_convert_turns)
    p = sub.add_parser("finalize", help="streaming dedup, closure, protected scan, validation and manifest")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--protected", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_finalize)
    p = sub.add_parser("validate")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--schema", action="store_true", help="also apply the JSON schema (needs jsonschema)")
    p.add_argument("--report", type=Path)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("audit-splits", help="stream-check split-group conflicts and protected-set placement")
    p.add_argument("inputs", nargs="+", help="record JSONL files or directories containing *.port-records.jsonl")
    p.add_argument("--protected", type=Path, required=True,
                   help="nonempty protected JSON containing question_hashes and ids")
    p.add_argument("--report", type=Path, required=True, help="write the content-pinned audit receipt here")
    p.set_defaults(func=cmd_audit_splits)
    p = sub.add_parser("dedup")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_dedup)
    p = sub.add_parser("protected")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "protected" / "bgkit-benchmarks.protected.json")
    p.set_defaults(func=cmd_protected)
    p = sub.add_parser("close")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--links", type=Path)
    p.add_argument("--drop", type=Path)
    p.add_argument("--protected", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_close)
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
