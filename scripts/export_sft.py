#!/usr/bin/env python3
"""Turn the reference corpus into supervised pairs of text, rendered by the model's own chat template exactly as at
inference (the llama.cpp server's /apply-template with the official template and the turn's tools).

  scripts/export_sft.py data/external_pilot/direct-core-v2-traces.jsonl.gz data/direct-core-v2.sft.jsonl --server http://127.0.0.1:8080 [--limit N] [--every K]
Each output line: {"id", "family", "skill", "prompt", "completion"}; train on the completion only.
"""
import argparse, hashlib, json, sys, time, urllib.error, urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.native import _strip_private
from natlang.corpus import program_id
from natlang.terminal import reply_only_sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--every", type=int, default=1, help="keep every K-th sample")
    ap.add_argument("--workers", type=int, default=1, help="bounded parallel template requests")
    ap.add_argument("--resume", action="store_true", help="append after verifying the last existing ID")
    ap.add_argument("--include-template", action="store_true",
                    help="include programs with provisional generated-text gold (excluded by default)")
    ap.add_argument("--end-token", default="<|im_end|>",
                    help="assistant message terminator used by the selected model template")
    ap.add_argument("--template-id", default="qwen-chatml",
                    help="human-readable identity of the server's chat template")
    a = ap.parse_args()
    if a.workers < 1 or a.every < 1 or a.limit < 1:
        ap.error("workers, every and limit must be positive")

    try:
        props = json.loads(urllib.request.urlopen(a.server + "/props", timeout=10).read())
    except (urllib.error.URLError, ValueError):
        props = {}
    template_source = props.get("chat_template_tool_use") or props.get("chat_template")
    template_hash = hashlib.sha256(template_source.encode()).hexdigest() if isinstance(template_source, str) else None
    renderer = {"version": "llama.cpp-apply-template/2", "template_id": a.template_id,
                "template_sha256": template_hash, "end_token": a.end_token,
                "server": a.server, "include_template": a.include_template,
                "terminal_tool_policy": "empty-success-turn-v2",
                "teacher_reasoning_policy": "render-if-present-v1"}
    manifest_path = a.dst.with_suffix(a.dst.suffix + ".manifest.json")

    def render(messages, tools):
        req = urllib.request.Request(a.server + "/apply-template", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": messages, "tools": _strip_private(tools)}).encode())
        for attempt in range(5):
            try:
                return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt"]
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                if attempt == 4:
                    raise
                time.sleep(min(0.25 * 2 ** attempt, 4))

    if renderer["template_sha256"] is None:
        probe = render([{"role": "system", "content": "template identity probe"},
                        {"role": "user", "content": "probe"}], [])
        renderer["template_sha256"] = hashlib.sha256(probe.encode()).hexdigest()
    if a.resume and a.dst.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text())["renderer"] != renderer:
            ap.error("resume renderer differs from existing SFT manifest")

    import gzip

    def lines():                                   # a .jsonl file, a .jsonl.gz file, or a directory of shards
        files = sorted(a.src.glob("part-*.jsonl.gz")) if a.src.is_dir() else [a.src]
        for path in files:
            with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
                yield from f

    def convert(s):
        s = reply_only_sample(s)
        prompt = render(s["messages"], s["tools"])
        calls = s["target"].get("tool_calls")
        reasoning = s.get("teacher_reasoning")
        if calls or reasoning:        # let the model template serialize both decisions and thinking
            tgt = {**s["target"]}
            if calls:
                tgt["tool_calls"] = [{**c, "id": f"x{j}"} for j, c in enumerate(calls)]
            if reasoning:
                tgt["reasoning_content"] = reasoning
            after = ([{"role": "tool", "tool_call_id": f"x{j}", "content": "X"}
                      for j in range(len(calls))] if calls else [{"role": "user", "content": "X"}])
            full = render(s["messages"] + [tgt] + after, s["tools"])
            if not full.startswith(prompt):
                raise ValueError("template does not preserve the assistant prefix; choose a compatible renderer")
            suffix = full[len(prompt):]
            if a.end_token not in suffix:
                raise ValueError("assistant end token is absent from rendered target")
            completion = suffix.split(a.end_token, 1)[0] + a.end_token
            if reasoning and reasoning not in completion:
                raise ValueError("template dropped teacher reasoning; use a reasoning-capable renderer")
        else:
            completion = s["target"]["content"] + a.end_token
        source_groups = s.get("source_groups") or []
        grouped = s.get("family") in ("lambda_scenario", "teacher_program") or s.get("kind") == "review"
        return {"id": s["id"], "program_id": source_groups[0] if grouped and source_groups else program_id(s),
                "source_groups": source_groups, "family": s.get("family", s.get("kind")),
                "skill": s["skill"], "renderer": a.template_id,
                "context_items": len(s["messages"]),
                "teacher_trajectory_id": s.get("teacher_trajectory_id"),
                "teacher_trajectory_digest": s.get("teacher_trajectory_digest"),
                "training_admission": s.get("training_admission"),
                "prompt": prompt, "completion": completion}

    existing = 0
    last_id = None
    if a.resume and a.dst.exists():
        with a.dst.open() as stream:
            for line in stream:
                last_id = json.loads(line)["id"]
                existing += 1

    def pending():
        selected = 0
        matched = existing == 0
        eligible = 0
        for line in lines():
            s = json.loads(line)
            if (s.get("provisional_gold") or s.get("template")) and not a.include_template:
                continue
            if eligible % a.every:
                eligible += 1
                continue
            eligible += 1
            if selected >= a.limit:
                break
            if selected < existing:
                if selected == existing - 1:
                    if s["id"] != last_id:
                        raise ValueError("resume destination does not match source at last ID")
                    matched = True
                selected += 1
                continue
            selected += 1
            yield s
        if not matched:
            raise ValueError("resume destination has more rows than source")

    n = existing
    items = iter(pending())
    with ThreadPoolExecutor(max_workers=a.workers) as pool, a.dst.open("a" if a.resume else "w") as out:
        futures = deque()
        for _ in range(a.workers * 4):
            try:
                futures.append(pool.submit(convert, next(items)))
            except StopIteration:
                break
        while futures:
            result = futures.popleft().result()
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1
            if n % 2000 == 0:
                print(n, flush=True)
            try:
                futures.append(pool.submit(convert, next(items)))
            except StopIteration:
                pass
    print(f"{n} pairs -> {a.dst}")
    manifest_path.write_text(json.dumps({"renderer": renderer, "source": str(a.src),
                                         "pairs": n, "every": a.every, "limit": a.limit}, indent=2) + "\n")


if __name__ == "__main__":
    main()
