#!/usr/bin/env python3
"""Decode throughput of one fixed recorded request, with and without an adapter, alternating (S7_RL.md §8.3).

usage: s7_decode_probe.py URL ROLLOUTS.jsonl ADAPTER_ID OUT.json [--reps 3] [--max-tokens 256]
Greedy, non-streaming; tok/s = completion_tokens / wall seconds of the request.
"""
import argparse
import json
import time
import urllib.request


def request(url: str, body: dict) -> tuple[float, dict]:
    data = json.dumps(body).encode()
    start = time.monotonic()
    req = urllib.request.Request(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        reply = json.loads(response.read())
    return time.monotonic() - start, reply


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url"); p.add_argument("rollouts"); p.add_argument("adapter"); p.add_argument("out")
    p.add_argument("--reps", type=int, default=3); p.add_argument("--max-tokens", type=int, default=256)
    a = p.parse_args()
    turn = next(t for line in open(a.rollouts) if line.strip() for t in json.loads(line).get("turns") or [])
    base = {"model": "lfm", "messages": turn["messages"], "max_tokens": a.max_tokens, "temperature": 0}
    if turn.get("tools"):
        base["tools"] = turn["tools"]
    runs = []
    for rep in range(a.reps):
        for arm in ("base", "adapter"):
            body = dict(base, **({"x_natlang_adapters": [{"id": a.adapter, "scale": 1}]} if arm == "adapter" else {}))
            seconds, reply = request(a.url, body)
            tokens = (reply.get("usage") or {}).get("completion_tokens") or 0
            runs.append({"rep": rep, "arm": arm, "seconds": seconds, "completion_tokens": tokens,
                         "prompt_tokens": (reply.get("usage") or {}).get("prompt_tokens"),
                         "tok_per_s": tokens / seconds if seconds else None})
    summary = {arm: sum(r["completion_tokens"] for r in runs if r["arm"] == arm) /
               sum(r["seconds"] for r in runs if r["arm"] == arm) for arm in ("base", "adapter")}
    result = {"schema": "natlang.s7-decode-probe/1", "runs": runs, "tok_per_s": summary,
              "adapter_slowdown": summary["base"] / summary["adapter"] if summary["adapter"] else None}
    open(a.out, "w").write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tok_per_s": summary, "adapter_slowdown": result["adapter_slowdown"]}))


if __name__ == "__main__":
    main()
