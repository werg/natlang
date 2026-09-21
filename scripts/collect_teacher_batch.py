#!/usr/bin/env python3
"""Collect frozen teacher programs as resumable, atomic, parallel jobs.

Each source row owns one result and trace file.  A completed result is reused
only when its source, model, seed, prompt, and continuation settings still
match.  The merged JSONL is rebuilt in source order after every completed job,
so interrupting the process loses at most the requests currently in flight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.decoder import LlamaServerDecoder
from natlang.native import NativeCallDecoder
from natlang.invocation import RunOptions, SeedPolicy
from scripts.collect_scenario_teacher import collect
from scripts.program_ir import digest, validate

VERSION = "natlang.teacher_batch/1"


def load_records(path: Path, start: int, limit: int | None) -> list[tuple[int, dict]]:
    records = []
    with path.open() as source:
        for index, raw in enumerate(source):
            if index < start:
                continue
            if limit is not None and len(records) >= limit:
                break
            record = json.loads(raw)
            validate(record)
            records.append((index, record))
    if limit is not None and len(records) != limit:
        raise ValueError("requested source range exceeds the frozen batch")
    return records


def job_key(index: int, record: dict) -> str:
    return f"{index:06d}-{digest(record)[:16]}"


def expected_provenance(record: dict, *, model_id: str, root_seed: int,
                        system_prompt: str, segment_turns: int,
                        segment_messages: int, cache_stable_tools: bool = False,
                        decode: str = "server", require_call: bool = False) -> dict:
    provenance = {"program_ir_sha256": digest(record), "model": model_id,
                  "seed_policy": vars(SeedPolicy("derived", root_seed)),
                  "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
                  "segment_turns": segment_turns, "segment_messages": segment_messages}
    if cache_stable_tools:
        provenance["cache_stable_tools"] = True
    if decode != "server":
        provenance["decode"] = decode
    if require_call:
        provenance["require_call"] = True
    return provenance


def result_matches(path: Path, record: dict, expected: dict) -> bool:
    try:
        row = json.loads(path.read_text())
        provenance = row["provenance"]
        return (row["task"]["program_ir"]["id"] == record["id"] and
                all(provenance.get(key) == value for key, value in expected.items()))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def write_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def merge_completed(records: list[tuple[int, dict]], jobs: Path, output: Path,
                    expected_for) -> tuple[int, list[int]]:
    rows, missing = [], []
    for index, record in records:
        path = jobs / (job_key(index, record) + ".result.json")
        if result_matches(path, record, expected_for(record)):
            rows.append(path.read_text())
        else:
            missing.append(index)
    temporary = output.with_suffix(output.suffix + ".building")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("".join(rows))
    temporary.replace(output)
    return len(rows), missing


def import_completed(paths: list[Path], records: list[tuple[int, dict]], jobs: Path,
                     expected_for, *, segment_turns: int, segment_messages: int) -> tuple[int, int]:
    """Adopt compatible older rows; reject pre-compaction conversations."""
    by_digest = {digest(record): (index, record) for index, record in records}
    imported = rejected = 0
    for source in paths:
        for raw in source.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            identity = row.get("provenance", {}).get("program_ir_sha256")
            matched = by_digest.get(identity)
            if not matched:
                continue
            index, record = matched
            expected = expected_for(record)
            provenance = row.get("provenance", {})
            base = {key: value for key, value in expected.items()
                    if key not in ("segment_turns", "segment_messages")}
            checkpoints = [turn for turn in row.get("trajectory", [])
                           if turn.get("phase") == "checkpoint"]
            compact = max((len(turn.get("context") or []) for turn in row.get("trajectory", [])), default=0)
            compatible = (all(provenance.get(key) == value for key, value in base.items()) and
                compact <= segment_messages * 2 and
                all(turn.get("segment_turns") == segment_turns and
                    turn.get("segment_messages") == segment_messages for turn in checkpoints))
            if not compatible:
                rejected += 1
                continue
            row["provenance"].update({"segment_turns": segment_turns,
                                      "segment_messages": segment_messages,
                                      "imported_from": str(source)})
            result = jobs / (job_key(index, record) + ".result.json")
            if not result_matches(result, record, expected):
                write_atomic(result, row)
                imported += 1
    return imported, rejected


def decoder_for(args) -> LlamaServerDecoder:
    if args.decode == "native":
        return NativeCallDecoder(
            args.server, timeout=args.request_timeout,
            write_constraints="runtime",
            allow_reply=not args.require_call,
            cache_stable_tools=getattr(args, "cache_stable_tools", False))
    chat_extra = {"thinking_budget_tokens": args.thinking_tokens, "top_p": 0.95,
                  "top_k": 20,
                  "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort}}
    if args.require_call:
        chat_extra["tool_choice"] = "required"
    return LlamaServerDecoder(
        args.server,
        timeout=args.request_timeout,
        chat_extra=chat_extra,
        tool_aliases={"call": args.call_tool_name}, json_text_values=True,
        cache_stable_tools=getattr(args, "cache_stable_tools", False))


def run_job(index: int, record: dict, args, system_prompt: str,
            jobs: Path, expected: dict) -> tuple[int, dict]:
    key = job_key(index, record)
    result = jobs / (key + ".result.json")
    if result_matches(result, record, expected):
        return index, json.loads(result.read_text())
    # A stale or interrupted artifact cannot resume execution and must not
    # accumulate beside the only result that the pipeline can consume.
    if result.exists():
        result.unlink()
    for stale in jobs.glob(key + ".stale*.json"):
        stale.unlink()
    for interrupted in [jobs / (key + ".trace.jsonl"), *jobs.glob(key + ".retry*.trace.jsonl")]:
        interrupted.unlink(missing_ok=True)
    trace = jobs / (key + ".trace.jsonl")
    run_id = hashlib.sha256(json.dumps({"batch": VERSION, "index": index,
        **expected}, sort_keys=True).encode()).hexdigest()[:32]
    row, _ = collect(record, decoder_for(args), model_id=args.model_id,
                     options=RunOptions(seed=SeedPolicy("derived", args.root_seed), run_id=run_id),
                     system_prompt=system_prompt, trace_path=trace,
                     segment_turns=args.segment_turns,
                     segment_messages=args.segment_messages)
    row["provenance"].update({"segment_turns": args.segment_turns,
                              "segment_messages": args.segment_messages})
    if getattr(args, "cache_stable_tools", False):
        row["provenance"]["cache_stable_tools"] = True
    decode = getattr(args, "decode", "server")
    if decode != "server":
        row["provenance"]["decode"] = decode
    if getattr(args, "require_call", False):
        row["provenance"]["require_call"] = True
    write_atomic(result, row)
    (jobs / f"{index:06d}.error.json").unlink(missing_ok=True)
    return index, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--segment-turns", type=int, default=6)
    parser.add_argument("--segment-messages", type=int, default=12)
    parser.add_argument("--thinking-tokens", type=int, default=256)
    parser.add_argument("--decode", choices=("server", "native"), default="server")
    parser.add_argument("--require-call", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--call-tool-name", default="call_function",
                        help="server-native name for the runtime call tool")
    parser.add_argument("--cache-stable-tools", action="store_true",
                        help="present state-dependent path and line constraints as stable base types")
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--system-file", type=Path,
                        default=Path("natlang/prompts/tools_teacher_compact.md"))
    parser.add_argument("--import-ir", type=Path, action="append", default=[],
                        help="adopt matching compact trajectories from an older collector; may repeat")
    args = parser.parse_args()
    if (args.start < 0 or args.workers < 1 or args.segment_turns < 1 or
            args.segment_messages < 5 or args.thinking_tokens < 0 or
            (args.limit is not None and args.limit < 1)):
        parser.error("invalid range, worker, continuation, or thinking setting")
    records = load_records(args.ir, args.start, args.limit)
    system_prompt = args.system_file.read_text()
    args.jobs.mkdir(parents=True, exist_ok=True)
    expected_for = lambda record: expected_provenance(
        record, model_id=args.model_id, root_seed=args.root_seed,
        system_prompt=system_prompt, segment_turns=args.segment_turns,
        segment_messages=args.segment_messages, cache_stable_tools=args.cache_stable_tools,
        decode=args.decode, require_call=args.require_call)
    if args.import_ir:
        imported, rejected = import_completed(args.import_ir, records, args.jobs, expected_for,
            segment_turns=args.segment_turns, segment_messages=args.segment_messages)
        print(f"import: {imported} compatible rows adopted; {rejected} legacy/incompatible rows rejected",
              flush=True)
    pending = [(index, record) for index, record in records
               if not result_matches(args.jobs / (job_key(index, record) + ".result.json"),
                                     record, expected_for(record))]
    completed, missing = merge_completed(records, args.jobs, args.out, expected_for)
    print(f"resume: {completed}/{len(records)} complete; {len(pending)} queued", flush=True)
    stopping = threading.Event()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    try:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="teacher") as pool:
            active = {}
            cursor = iter(pending)
            while not stopping.is_set():
                while len(active) < args.workers:
                    try:
                        index, record = next(cursor)
                    except StopIteration:
                        break
                    future = pool.submit(run_job, index, record, args, system_prompt,
                                         args.jobs, expected_for(record))
                    active[future] = (index, record["id"])
                if not active:
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    index, program_id = active.pop(future)
                    try:
                        _, row = future.result()
                        print(f"{index} {program_id}: {row['outcome']['status']} "
                              f"accepted={row['outcome']['accepted']}", flush=True)
                    except Exception as exc:
                        error = {"index": index, "program_id": program_id,
                                 "error": f"{type(exc).__name__}: {exc}"}
                        write_atomic(args.jobs / f"{index:06d}.error.json", error)
                        print(f"{index} {program_id}: ERROR {error['error']}",
                              file=sys.stderr, flush=True)
                    completed, missing = merge_completed(records, args.jobs, args.out, expected_for)
                    print(f"checkpoint: {completed}/{len(records)} complete", flush=True)
    except KeyboardInterrupt:
        stopping.set()
        print("interrupted; completed jobs are durable", file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
    completed, missing = merge_completed(records, args.jobs, args.out, expected_for)
    manifest = {"version": VERSION, "source": str(args.ir),
                "source_sha256": hashlib.sha256(args.ir.read_bytes()).hexdigest(),
                "range": {"start": args.start, "count": len(records)},
                "model": args.model_id, "root_seed": args.root_seed,
                "decode": args.decode, "cache_stable_tools": args.cache_stable_tools,
                "require_call": args.require_call,
                "segment_turns": args.segment_turns, "segment_messages": args.segment_messages,
                "workers": args.workers, "completed": completed, "missing": missing,
                "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    write_atomic(args.out.with_suffix(args.out.suffix + ".manifest.json"), manifest)
    print(f"final: {completed}/{len(records)} complete -> {args.out}", flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
