#!/usr/bin/env python3
"""Run conformance programs against the served model through structured tools."""
import re, argparse, json, sys, time, os
from collections import Counter
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.decoder import LlamaServerDecoder
from natlang.host import load
from natlang.surface import ToolSurface
from natlang.tool_agent import ToolAgent
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.values import dump
from natlang.corpus import file_digest

ap = argparse.ArgumentParser()
ap.add_argument("--review-prompt", choices=["baseline", "repeat_instructions", "checklist"], default="baseline")
ap.add_argument("--decode", default="native", choices=("native", "server"),
                help="native: our grammar over the model's native call text; server: the server's tool calling")
ap.add_argument("--require-call", action="store_true", help="native only: no reply until a call was made")
ap.add_argument("--verbose", action="store_true", help="print every action as it happens")
ap.add_argument("--max-episodes", type=int, default=64)
ap.add_argument("--server", default="http://127.0.0.1:8080")
ap.add_argument("--thinking", type=int, default=None, help="thinking budget in tokens (reasoning teachers)")
ap.add_argument("--timeout", type=float, default=600)
ap.add_argument("--temperature", type=float, default=0.2)
ap.add_argument("--system-file", type=Path, default=None, help="system prompt for the tool surface (per-model opt-in)")
ap.add_argument("--judge-server", default="http://127.0.0.1:8081", help="model that answers judge checks; 'none' to skip")
ap.add_argument("--alias", action="append", default=[], help="tool renames for this model's server, e.g. call=call_function")
ap.add_argument("--out", type=Path, help="machine-readable results; defaults to runs/baseline-<timestamp>.json")
ap.add_argument("--model-label", default="unspecified")
ap.add_argument("--validation-feedback", choices=("local", "caller"), default=None)
ap.add_argument("--careful-threshold", type=float)
ap.add_argument("--state-view", action="store_true", help="experimental expanded execution state")
ap.add_argument("--review-scope", choices=["values", "actions"], default="values")
ap.add_argument("--withdrawal-policy", choices=["caller", "retry"], default="caller")
ap.add_argument("--write-constraints", choices=("typed", "runtime"), default="runtime")
ap.add_argument("ids", nargs="*")
a = ap.parse_args()
a.validation_feedback = a.validation_feedback or "caller"
root = Path(__file__).resolve().parent.parent
files = sorted((root / "conformance" / "programs").glob("*.yaml"))
files = [f for f in files if not a.ids or any(f.stem.startswith(i) for i in a.ids)]
extra = {} if a.thinking is None else {"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20}
dec = (NativeCallDecoder(a.server, timeout=a.timeout, write_constraints=a.write_constraints) if a.decode == "native"
       else LlamaServerDecoder(a.server, timeout=a.timeout, chat_extra=extra,
                               tool_aliases=dict(x.split("=", 1) for x in a.alias)))
from natlang.checks import grade, make_judge
judge = None if a.judge_server == "none" else make_judge(
    LlamaServerDecoder(a.judge_server, timeout=a.timeout, chat_extra={"chat_template_kwargs": {"enable_thinking": False}}))
print(f"surface=tools decode={a.decode}")
records = []
for f in files:
    doc = yaml.safe_load(f.read_text())
    if ("program" not in doc and "program_file" not in doc) or "streams" in doc:
        continue
    class Live(list):
        def append(self, x):
            super().append(x)
            if a.verbose:
                print(f"      {x['kind']:<9}{x['action'][:150]}", flush=True)
    log = Live()
    if a.verbose:
        print(f"  > {f.stem}", flush=True)
    make = lambda lam: ToolAgent(dec, temperature=a.temperature, log=log,
                                 validation_feedback=a.validation_feedback, careful_threshold=a.careful_threshold, surface=ToolSurface(state_view=a.state_view), review_scope=a.review_scope, withdrawal_policy=a.withdrawal_policy, review_prompt=a.review_prompt,
                                 **({"system_prompt": a.system_file.read_text()} if a.system_file else {}))
    rt = Runtime(make, max_episodes=a.max_episodes)
    t = time.time()
    try:
        src = (f.parent / doc["program_file"]) if doc.get("program_file") else f
        out, value = rt.run_root(load(src, doc.get("inputs") or {}))
        kind = out.kind
    except Exception as e:
        import traceback
        out, kind, value = None, f"crash: {type(e).__name__}: {e}"[:60], None
        if a.verbose:
            traceback.print_exc()
    try:
        verdict, why = grade(doc["expect"], kind, dump(value) if kind == "done" else None,
                             note=getattr(out, "detail", "") or "", judge=judge, emitted=rt.emitted)
    except OSError as e:
        verdict, why = "?", [f"judge unavailable: {e}"]
    correct = verdict == "yes"
    shapes = sorted({m.group(1) for l in log if l["kind"] == "ok"
                     for m in [re.search(r'^(call) ', l["action"])] if m} |
                    {m.group(1) for l in log if l["kind"] in ("ok", "done")
                     for m in [re.search(r'"function": "([\w/]+)"', l["action"])] if m})
    rejected = sum(1 for l in log if l["kind"] in ("rejected", "error"))
    first = log[0]["action"].splitlines()[0][:60] if log else ""
    print(f"{f.stem:<32} {kind:<10} correct={verdict:<3} actions={len(log):<3} "
          f"rejected={rejected:<3} episodes={rt.episodes_started:<3} {time.time()-t:5.1f}s  structure={','.join(shapes) or '-':<12} first: {first}"
          + ("".join(f"\n      failed: {w}" for w in why)))
    records.append({"program": f.stem, "program_sha256": file_digest(f), "status": kind,
                    "verdict": verdict, "details": why, "value": dump(value) if kind == "done" else None,
                    "emitted": rt.emitted, "actions": len(log), "rejected": rejected,
                    "episodes": rt.episodes_started, "seconds": time.time() - t})
counts = dict(Counter(r["verdict"] for r in records))
print(f"\nprograms={len(records)} correct={counts.get('yes', 0)} incorrect={counts.get('no', 0)} "
      f"unjudged={counts.get('?', 0)}")
result_path = a.out or root / "runs" / f"baseline-{time.time_ns()}.json"
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps({"model": a.model_label, "server": a.server, "surface": "tools",
                                  "decode": a.decode, "marks": os.environ.get("NATLANG_MARKS", "1"),
                                  "validation_feedback": a.validation_feedback, "careful_threshold": a.careful_threshold, "state_view": a.state_view, "review_scope": a.review_scope, "review_prompt": a.review_prompt, "withdrawal_policy": a.withdrawal_policy,
                                  "write_constraints": a.write_constraints,
                                  "done_arg": os.environ.get("NATLANG_DONE_ARG", "1"),
                                  "counts": counts, "programs": records, "usage": dec.usage}, indent=2) + "\n")
print(f"results: {result_path}")
if a.decode == "native" and dec.stats["p_call_first"]:
    pc = [p for p in dec.stats["p_call_first"] if p is not None]
    print(f"\nturns={dec.stats['turns']} tool-call turns={dec.stats['turns']-dec.stats['replies']} replies={dec.stats['replies']} "
          f"mean P(model starts a call)={sum(pc)/len(pc) if pc else 'unavailable'}")
u = dec.usage
if u["turns"]:
    print(f"server usage: {u['turns']} turns, {u['completion_tokens']} completion tokens, {u['seconds']:.0f}s "
          f"({u['completion_tokens']/max(u['seconds'],1e-9):.0f} tok/s incl. prefill)")
