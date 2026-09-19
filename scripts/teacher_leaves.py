#!/usr/bin/env python3
"""Teacher-written references for generative leaves (PLAN 10.1): the teacher performs the leaf through the real
harness; the output is kept only if it passes the leaf's checks (crisp constraints plus a judge question). Accepted
outputs go to data/leaf_references.jsonl, keyed by function and arguments; the generators use them on the next run.

  scripts/teacher_leaves.py --families cb_shopkeeper cb_webserver --n 12 --seed 31 --server http://127.0.0.1:8081
Use the same --seed and families as the corpus run whose leaves should be covered.
"""
import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from generate import run_program                                             # noqa: E402
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
                      {"kind": "judge", "answer": True, "question":
                          f"A shopkeeper decided on this action: {json.dumps(a['action'])}. Is the line consistent with it, "
                          "promising no goods, quantities or prices other than those in the action?"}],
    "page_content": lambda a: [NO_MARKUP_ATTACK, {"kind": "crisp", "code": "wordCount(value) >= 15 && wordCount(value) <= 320"},
                               {"kind": "judge", "answer": True, "question": f"Is this HTML fragment a fitting page for this purpose: {a['purpose']}"}]
                              + [{"kind": "crisp", "code": f"value.includes({json.dumps(e['message'][:20])})"} for e in a["entries"][-1:]
                                 if "recent" in a["purpose"] or "entries" in a["purpose"]],
    "submission_page": lambda a: [NO_MARKUP_ATTACK, {"kind": "crisp", "code": "value.includes('href=\"/\"') && wordCount(value) <= 120"},
                                  {"kind": "judge", "answer": True, "question":
                                      "Does this message tell the visitor that their submission was " +
                                      ("accepted" if a["review"]["accept"] else "NOT accepted, and why") + "?"}],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=["cb_shopkeeper", "cb_webserver"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--thinking", type=int, default=256)
    ap.add_argument("--limit", type=int, default=10**9, help="stop after this many leaves")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for i in range(a.n):                                # collect the leaves these programs need
        run_program(C.CODEBASES[a.families[i % len(a.families)]](rng), check_grammar=False)
    todo, seen = [], set()
    for fn, args in C.MISSES:
        k = C.ref_key(fn, args)
        if k not in seen and k not in C.REFERENCES:
            seen.add(k); todo.append((k, fn, args))
    print(f"{len(todo)} generative leaves without a reference", flush=True)
    dec = LlamaServerDecoder(a.server, timeout=900, chat_extra={"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20},
                             tool_aliases={"call": "call_function"})
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
        out, value = Runtime(lambda lam: ToolAgent(dec, temperature=0.7), max_episodes=4).run_root(root)
        text = dump(value) if out.kind == "done" else None
        results = run_checks(CHECKS[fn](args), text, judge) if text else []
        ok = bool(text) and all(r is True for _, r in results)
        if ok:
            kept += 1
            with C.REF_FILE.open("a") as f:
                f.write(json.dumps({"key": k, "function": fn, "args": args, "value": text}, ensure_ascii=False) + "\n")
        failed = [c.get("code") or c.get("question", "")[:60] for c, r in results if r is not True]
        print(f"[{fn}] {'KEEP' if ok else 'drop'} {time.time() - t0:4.0f}s  {(text or out.kind)[:90]!r}  {failed if failed else ''}", flush=True)
    print(f"kept {kept} of {min(len(todo), a.limit)}; usage {dec.usage}")


if __name__ == "__main__":
    main()
