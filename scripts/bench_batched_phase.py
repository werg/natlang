"""Gate 2 analysis: wall span, throughput and per-call latency of one definition's calls in a trace directory.

    python3 scripts/bench_batched_phase.py TRACE_DIR RUN_PREFIX [DEFINITION]

RUN_PREFIX is the run's trace file prefix (task-N-UUID). Each call's start is its manifest `observed_at`; its latency is
the `elapsed_ms` of its last event. Model-request events carry the scheduler's batch records (`batch_size`,
`in_flight`, `queue_wait_ms`), summarised as occupancy.
"""
import datetime
import glob
import json
import sys

directory, prefix = sys.argv[1], sys.argv[2]
definition = sys.argv[3] if len(sys.argv) > 3 else "classify"
rows, batch, in_flight, waits, requests = [], [], [], [], 0
for path in glob.glob(f"{directory}/{prefix}_*.jsonl"):
    events = [json.loads(line) for line in open(path)]
    manifest = events[0]
    if manifest.get("definition_name") != definition:
        continue
    start = datetime.datetime.fromisoformat(manifest["observed_at"].replace("Z", "+00:00"))
    elapsed = events[-1].get("elapsed_ms", 0)
    rows.append((start, start + datetime.timedelta(milliseconds=elapsed), elapsed))
    for event in events:
        if event["kind"] == "model_request" and "batch_size" in event:
            requests += 1
            batch.append(event["batch_size"])
            in_flight.append(event.get("in_flight", 0))
            waits.append(event.get("queue_wait_ms", 0))
rows.sort()
latency = sorted(row[2] for row in rows)
pct = lambda values, p: values[min(len(values) - 1, int(p * len(values)))]
span = (max(row[1] for row in rows) - min(row[0] for row in rows)).total_seconds()
mean = lambda values: round(sum(values) / len(values), 2) if values else None
print(json.dumps({
    "definition": definition, "calls": len(rows), "phase_span_s": round(span, 1), "calls_per_min": round(60 * len(rows) / span, 2),
    "latency_p50_s": round(pct(latency, 0.5) / 1000, 1), "latency_p95_s": round(pct(latency, 0.95) / 1000, 1),
    "model_requests": requests, "mean_batch_size": mean(batch), "mean_in_flight": mean(in_flight),
    "mean_queue_wait_ms": mean(waits)}))
