#!/usr/bin/env python3
"""Teacher-written references for generative leaves (PLAN 10.1): the teacher performs the leaf through the real
harness; the output is kept only if it passes the leaf's checks (crisp constraints plus a judge question). Accepted
outputs go to data/leaf_references.jsonl, keyed by function and arguments; the generators use them on the next run.

  scripts/teacher_leaves.py --ir data/external_pilot/synthetic-all-current.ir.jsonl --limit 12
  scripts/teacher_leaves.py --families cb_shopkeeper cb_webserver --n 12 --seed 31
Use --ir for a frozen corpus; seed replay is only for a newly generated corpus.
"""
import argparse, json, random, sys, time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from generate import MIXES, make_program, run_program                                             # noqa: E402
from natlang.checks import make_judge, run_checks                            # noqa: E402
from natlang.codebase import load_function                                   # noqa: E402
from natlang.decoder import LlamaServerDecoder                               # noqa: E402
from natlang.gen import codebases as C                                       # noqa: E402
from natlang.host import instantiate                                         # noqa: E402
from natlang.runtime import Runtime                                          # noqa: E402
from natlang.tool_agent import ToolAgent                                     # noqa: E402
from natlang.types import TypeEnv                                            # noqa: E402
from natlang.values import coerce, dump                                      # noqa: E402

WHERE = {"say": "shopkeeper/serve", "page_content": "webserver/handle", "submission_page": "webserver/handle"}   # the parent
NO_MARKUP_ATTACK = {"kind": "crisp", "code": "!/<\\s*script|<\\s*html|<\\s*body|javascript:/i.test(value)"}
CHECKS = {
    "say": lambda a: [{"kind": "crisp", "code": "wordCount(value) >= 2 && wordCount(value) <= 45"},
                      {"kind": "judge", "answer": True, "question": (
                          f"A shopkeeper sells {a['action']['qty']} {a['action']['good']} at {a['action']['price']} coins each "
                          f"(action: {a['action']['code']}). Does the line agree with that sale and promise nothing else?"
                          if a["action"]["code"].startswith("sell") else
                          f"A shopkeeper's action is '{a['action']['code']}' (no sale takes place"
                          + (f"; the price mentioned may only be {a['action']['price']}" if a["action"]["price"] else "")
                          + "). Does the line fit that action without promising or handing over any goods?")}],
    "page_content": lambda a: [NO_MARKUP_ATTACK, {"kind": "crisp", "code": "wordCount(value) >= 15 && wordCount(value) <= 320"},
                               {"kind": "judge", "answer": True, "question": f"Is this HTML fragment a fitting page for this purpose: {a['purpose']}"}]
                              + ([{"kind": "crisp", "code": f"value.includes({json.dumps(e['message'][:20])})"} for e in a["entries"][-1:]
                                 ] if "recent" in a["purpose"] or "entries" in a["purpose"] else []),
    "submission_page": lambda a: [NO_MARKUP_ATTACK, {"kind": "crisp", "code": "value.includes('href=\"/\"') && wordCount(value) <= 120"},
                                  {"kind": "judge", "answer": True, "question":
                                      "Does this message tell the visitor that their submission was " +
                                      ("accepted" if a["review"]["accept"] else "NOT accepted, and why") + "?"}],
}


def missing_from_ir(path: Path):
    """Collect the exact, distinct template cases in a frozen semantic corpus."""
    todo, seen = [], set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            oracles = record["semantics"].get("leaf_oracles", {})
            if not isinstance(oracles, dict):
                continue
            for fn, oracle in oracles.items():
                if fn not in WHERE:
                    continue
                for case in oracle.get("cases", []):
                    if not case.get("template"):
                        continue
                    args = case["input"]
                    key = C.ref_key(fn, args)
                    if key not in seen and key not in C.REFERENCES:
                        seen.add(key)
                        todo.append((key, fn, args))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", type=Path, help="collect exact missing template cases from a frozen program IR")
    ap.add_argument("--families", nargs="*", default=["cb_shopkeeper", "cb_webserver"])
    ap.add_argument("--mix", default=None, help="a mix of scripts/generate.py, instead of --families")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--thinking", type=int, default=256)
    ap.add_argument("--limit", type=int, default=10**9, help="stop after this many leaves")
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="low")
    ap.add_argument("--system-file", type=Path, default=ROOT / "natlang/prompts/tools_teacher_compact.md")
    ap.add_argument("--dry-run", action="store_true", help="audit results without appending references")
    ap.add_argument("--audit-out", type=Path, help="all accepted and rejected attempts, with checks and traces")
    a = ap.parse_args()
    audit_path = a.audit_out or ROOT / "runs" / f"teacher-leaves-audit-{time.time_ns()}.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists():
        ap.error(f"refusing overwrite: {audit_path}")
    prompt = a.system_file.read_text()
    with urllib.request.urlopen(a.server.rstrip('/') + '/v1/models', timeout=30) as response:
        model_metadata = json.load(response)
    if a.ir:
        todo = missing_from_ir(a.ir)
    else:
        families = MIXES[a.mix] if a.mix else a.families
        for i in range(a.n):                                # collect the leaves these programs need: the same programs
            fam, prog = make_program(a.seed, i, families)   # as `generate.py --seed S` makes, by construction
            if fam in ("cb_shopkeeper", "cb_webserver"):
                run_program(prog, check_grammar=False)
        todo, seen = [], set()
        for fn, args in C.MISSES:
            k = C.ref_key(fn, args)
            if k not in seen and k not in C.REFERENCES:
                seen.add(k); todo.append((k, fn, args))
    print(f"{len(todo)} generative leaves without a reference", flush=True)
    dec = LlamaServerDecoder(a.server, timeout=900, chat_extra={"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20,
                                         "chat_template_kwargs": {"reasoning_effort": a.reasoning_effort}},
                             tool_aliases={"call": "call_function"}, json_text_values=True)
    judge = make_judge(LlamaServerDecoder(a.server, timeout=300, chat_extra={"chat_template_kwargs": {"enable_thinking": False}}))
    C.REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    for k, fn, args in todo[: a.limit]:
        t0 = time.time()
        definition = load_function(ROOT / "codebases" / (WHERE[fn] + ".nl")).codebase[fn]    # with the types it inherits
        root = instantiate(definition)
        env = root.env(TypeEnv())
        for name, value in args.items():
            root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
        log, transcript = [], []
        out, value = Runtime(lambda lam: ToolAgent(dec, temperature=a.temperature, system_prompt=prompt,
                             validation_feedback="caller", log=log, transcript=transcript), max_episodes=4).run_root(root)
        text = dump(value) if out.kind == "done" else None
        results = run_checks(CHECKS[fn](args), text, judge) if text else []
        ok = bool(text) and all(r is True for _, r in results)
        if ok:
            kept += 1
            if not a.dry_run:
                with C.REF_FILE.open("a") as f:
                    f.write(json.dumps({"key": k, "function": fn, "args": args, "value": text}, ensure_ascii=False) + "\n")
        with audit_path.open("a") as f:
            f.write(json.dumps({"key":k,"function":fn,"args":args,"value":text,"accepted":ok,
                               "admitted":ok and not a.dry_run,"status":out.kind,"detail":out.detail,
                               "checks":results,"log":log,"transcript":transcript,"system_prompt":prompt,
                               "model_metadata":model_metadata,"temperature":a.temperature,
                               "thinking":a.thinking,"reasoning_effort":a.reasoning_effort,
                               "json_text_values":True},ensure_ascii=False) + "\n")
        failed = [c.get("code") or c.get("question", "")[:60] for c, r in results if r is not True]
        print(f"[{fn}] {'KEEP' if ok else 'drop'} {time.time() - t0:4.0f}s  {(text or out.kind)[:90]!r}  {failed if failed else ''}", flush=True)
    print(f"passed {kept} of {min(len(todo), a.limit)}; admitted {0 if a.dry_run else kept}; usage {dec.usage}; audit {audit_path}")


if __name__ == "__main__":
    main()
