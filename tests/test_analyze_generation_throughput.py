import json

from scripts.analyze_generation_throughput import summarize


def test_summarize_pairs_requests_and_exposes_parallelism(tmp_path):
    traces = []
    for index, events in enumerate((
        [{"kind": "model_request", "phase": "start", "call_id": "a", "turn": 1, "elapsed_ms": 0},
         {"kind": "model_request", "phase": "end", "call_id": "a", "turn": 1,
          "elapsed_ms": 1000, "duration_ms": 1000, "prompt_tokens": 100, "completion_tokens": 10}],
        [{"kind": "model_request", "phase": "start", "call_id": "b", "turn": 1, "elapsed_ms": 0},
         {"kind": "model_request", "phase": "end", "call_id": "b", "turn": 1,
          "elapsed_ms": 1000, "duration_ms": 1000, "prompt_tokens": 90, "completion_tokens": 10}],
    )):
        path = tmp_path / f"{index}.trace.jsonl"
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        traces.append(path)
    report = summarize(traces)
    assert report["completed_requests"] == 2
    assert report["request_parallelism"] == 2
    assert report["completion_tokens_per_wall_second"] == 20
