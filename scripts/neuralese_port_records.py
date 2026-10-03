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
Converters reading parquet need pyarrow (for example /home/werg/bgkit/.venv/bin/python).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuralese_data import agents, bgkit, dedup, finalize, inventory, schnitzel, schnitzel_turns, splits, swe_compaction, trajectory  # noqa: E402
from neuralese_data.common import DEFAULT_OUTPUT_ROOT, Reject, Sink  # noqa: E402
from neuralese_data.records import validate, validate_with_schema  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _records(paths):
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def _record_files(inputs: list[str]) -> list[Path]:
    out = []
    for raw in inputs:
        p = Path(raw)
        out.extend(sorted(p.glob("*.port-records.jsonl")) if p.is_dir() else [p])
    return out


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


def leakage(record: dict) -> list[str]:
    """Writer inputs must not contain the target (S1 §6.5)."""
    value = record["target"]["value"]
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    consumer = "\n".join(m.get("content") or "" for m in record["consumer"]["context"])
    exact = {s.get("title") for s in record["sources"]} | {r["text"] for s in record["sources"] for r in s["exact_refs"]}
    if len(text.strip()) < 12 or text.strip() in consumer or text.strip() in exact:
        return []  # short answers, answers the consumer's request names, or exact source references
    writer = record["writer"]
    haystack = "\n".join([writer["instructions"], writer.get("instructions_general", ""),
                          *(m.get("content") or "" for m in writer["context"])])
    return ["target text appears in writer inputs"] if text.strip() in haystack else []


def cmd_validate(args):
    check = validate_with_schema if args.schema else validate
    totals, failures, ids = Counter(), Counter(), Counter()
    examples = {}
    for path in _record_files(args.inputs):
        for record in _records([path]):
            totals[path.name] += 1
            ids[record.get("id")] += 1
            errors = check(record) + leakage(record)
            for e in errors:
                key = e.split(":")[0]
                failures[key] += 1
                examples.setdefault(key, f"{record.get('id')}: {e}")
    dup_ids = {k: v for k, v in ids.items() if v > 1}
    report = {"files": dict(totals), "records": sum(totals.values()), "errors": dict(failures),
              "error_examples": examples, "duplicate_ids": len(dup_ids)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 1 if failures or dup_ids else 0


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
