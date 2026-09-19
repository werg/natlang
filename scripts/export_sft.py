#!/usr/bin/env python3
"""Turn the reference corpus into supervised pairs of text, rendered by the model's own chat template exactly as at
inference (the llama.cpp server's /apply-template with the official template and the turn's tools).

  scripts/export_sft.py data/ref-v5.jsonl data/sft-v5.jsonl --server http://127.0.0.1:8080 [--limit N] [--every K]
Each output line: {"id", "family", "skill", "prompt", "completion"}; train on the completion only.
"""
import argparse, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.native import _strip_private

END = "<|im_end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--every", type=int, default=1, help="keep every K-th sample")
    a = ap.parse_args()

    def render(messages, tools):
        req = urllib.request.Request(a.server + "/apply-template", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": messages, "tools": _strip_private(tools)}).encode())
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["prompt"]

    import gzip

    def lines():                                   # a .jsonl file, a .jsonl.gz file, or a directory of shards
        files = sorted(a.src.glob("part-*.jsonl.gz")) if a.src.is_dir() else [a.src]
        for path in files:
            with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
                yield from f

    n = 0
    with a.dst.open("w") as out:
        for i, line in enumerate(lines()):
            if i % a.every or n >= a.limit:
                continue
            s = json.loads(line)
            prompt = render(s["messages"], s["tools"])
            calls = s["target"].get("tool_calls")
            if calls:                      # the template decides how a call reads: that is what the history will show
                tgt = {**s["target"], "tool_calls": [{**c, "id": f"x{j}"} for j, c in enumerate(calls)]}
                after = [{"role": "tool", "tool_call_id": f"x{j}", "content": "X"} for j in range(len(calls))]
                full = render(s["messages"] + [tgt] + after, s["tools"])
                base = prompt[: prompt.rindex("<|im_start|>assistant")]
                completion = full[len(base):].split("<|im_start|>assistant\n", 1)[1].split(END)[0] + END
            else:
                completion = s["target"]["content"] + END
            out.write(json.dumps({"id": s["id"], "family": s["family"], "skill": s["skill"], "prompt": prompt,
                                  "completion": completion}, ensure_ascii=False) + "\n")
            n += 1
            if n % 2000 == 0:
                print(n, flush=True)
    print(f"{n} pairs -> {a.dst}")


if __name__ == "__main__":
    main()
