#!/usr/bin/env python3
"""Run conformance programs against the served model and summarize. Usage: baseline.py [--wrapper W] [IDS...]"""
import argparse, json, sys, time
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.decoder import WRAPPERS, LlamaServerDecoder
from natlang.host import load
from natlang.model_agent import SMALL_PROMPT, SYSTEM_PROMPT, ModelAgent
from natlang.tool_agent import ToolAgent
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.values import dump

ap = argparse.ArgumentParser()
ap.add_argument("--surface", default="tools", choices=("tools", "text"))
ap.add_argument("--decode", default="native", choices=("native", "server"),
                help="native: our grammar over the model's native call text; server: the server's tool calling")
ap.add_argument("--require-call", action="store_true", help="native only: no reply until a call was made")
ap.add_argument("--verbose", action="store_true", help="print every action as it happens")
ap.add_argument("--max-episodes", type=int, default=64)
ap.add_argument("--server", default="http://127.0.0.1:8080")
ap.add_argument("--thinking", type=int, default=None, help="thinking budget in tokens (reasoning teachers)")
ap.add_argument("--timeout", type=float, default=600)
ap.add_argument("--wrapper", default="generic", choices=tuple(WRAPPERS))
ap.add_argument("--prompt", default="small", choices=("small", "full"))
ap.add_argument("--temperature", type=float, default=0.2)
ap.add_argument("ids", nargs="*")
a = ap.parse_args()
root = Path(__file__).resolve().parent.parent
files = sorted((root / "conformance" / "programs").glob("*.yaml"))
files = [f for f in files if not a.ids or any(f.stem.startswith(i) for i in a.ids)]
extra = {} if a.thinking is None else {"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20}
dec = (NativeCallDecoder(a.server, timeout=a.timeout) if a.decode == "native"
       else LlamaServerDecoder(a.server, timeout=a.timeout, chat_extra=extra))
prompt = SMALL_PROMPT if a.prompt == "small" else SYSTEM_PROMPT
print(f"surface={a.surface} decode={a.decode}")
for f in files:
    doc = yaml.safe_load(f.read_text())
    if "program" not in doc or "streams" in doc:
        continue
    class Live(list):
        def append(self, x):
            super().append(x)
            if a.verbose:
                print(f"      {x['kind']:<9}{x['action'][:150]}", flush=True)
    log = Live()
    if a.verbose:
        print(f"  > {f.stem}", flush=True)
    if a.surface == "tools":
        make = lambda lam: ToolAgent(dec, temperature=a.temperature, log=log)
    else:
        make = lambda lam: ModelAgent(dec, wrapper=WRAPPERS[a.wrapper], temperature=a.temperature,
                                      system_prompt=prompt, log=log)
    rt = Runtime(make, max_episodes=a.max_episodes)
    t = time.time()
    try:
        out, value = rt.run_root(load(f, doc.get("inputs") or {}))
        kind = out.kind
    except Exception as e:
        kind, value = f"crash: {type(e).__name__}: {e}"[:60], None
    exp = doc["expect"]
    correct = ("value" in exp and kind == "done" and dump(value) == exp["value"]) or \
              ("status" in exp and kind == exp["status"])
    rejected = sum(1 for l in log if l["kind"] in ("rejected", "error"))
    first = log[0]["action"].splitlines()[0][:60] if log else ""
    print(f"{f.stem:<32} {kind:<10} correct={'yes' if correct else 'no ':<3} actions={len(log):<3} "
          f"rejected={rejected:<3} episodes={rt.episodes_started:<3} {time.time()-t:5.1f}s  first: {first}")
if a.decode == "native" and dec.stats["p_call_first"]:
    pc = dec.stats["p_call_first"]
    print(f"\nturns={dec.stats['turns']} tool-call turns={dec.stats['turns']-dec.stats['replies']} replies={dec.stats['replies']} "
          f"mean P(model starts a call)={sum(pc)/len(pc):.2f}")
u = dec.usage
if u["turns"]:
    print(f"server usage: {u['turns']} turns, {u['completion_tokens']} completion tokens, {u['seconds']:.0f}s "
          f"({u['completion_tokens']/max(u['seconds'],1e-9):.0f} tok/s incl. prefill)")
