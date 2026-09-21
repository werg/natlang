#!/usr/bin/env python3
"""Summarize model request throughput from one or more reduction traces."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    requests, starts = [], {}
    first = last = None
    for path in paths:
        for raw in path.open():
            event = json.loads(raw)
            if event.get("kind") != "model_request":
                continue
            elapsed = event.get("elapsed_ms")
            if elapsed is not None:
                first = elapsed if first is None else min(first, elapsed)
                last = elapsed if last is None else max(last, elapsed)
            identity = (str(path), event.get("call_id"), event.get("turn"), event.get("purpose"))
            if event.get("phase") == "start":
                starts[identity] = event
            elif event.get("phase") == "end":
                start = starts.pop(identity, {})
                duration = event.get("duration_ms")
                if duration is None and elapsed is not None and start.get("elapsed_ms") is not None:
                    duration = elapsed - start["elapsed_ms"]
                requests.append({"duration_ms": duration,
                    "prompt_tokens": event.get("prompt_tokens") or 0,
                    "completion_tokens": event.get("completion_tokens") or 0,
                    "checkpoint": event.get("purpose") == "checkpoint"})
    durations = [row["duration_ms"] for row in requests if row["duration_ms"] is not None]
    wall = max(0, (last or 0) - (first or 0)) / 1000
    prompt = sum(row["prompt_tokens"] for row in requests)
    completion = sum(row["completion_tokens"] for row in requests)
    return {"traces": len(paths), "completed_requests": len(requests),
            "incomplete_requests": len(starts), "checkpoint_requests": sum(r["checkpoint"] for r in requests),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "observed_wall_seconds": round(wall, 3),
            "summed_request_seconds": round(sum(durations) / 1000, 3),
            "request_parallelism": round(sum(durations) / 1000 / wall, 3) if wall else None,
            "requests_per_wall_minute": round(len(requests) * 60 / wall, 3) if wall else None,
            "tokens_per_wall_second": round((prompt + completion) / wall, 3) if wall else None,
            "completion_tokens_per_wall_second": round(completion / wall, 3) if wall else None,
            "latency_ms": ({"min": min(durations), "median": round(statistics.median(durations)),
                            "max": max(durations)} if durations else None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    paths = []
    for path in args.paths:
        paths.extend(sorted(path.glob("*.trace.jsonl")) if path.is_dir() else [path])
    print(json.dumps(summarize(paths), indent=2))


if __name__ == "__main__":
    main()
