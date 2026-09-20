#!/usr/bin/env python3
"""Teacher-written references for generative leaves (PLAN 10.1): the teacher performs the leaf through the real
harness; the output is kept only if it passes the leaf's checks (crisp constraints plus a judge question). Accepted
outputs go to data/leaf_references.jsonl, keyed by function and arguments; the generators use them on the next run.

  scripts/teacher_leaves.py --ir data/external_pilot/synthetic-all-current.ir.jsonl --limit 12
  scripts/teacher_leaves.py --families cb_shopkeeper cb_webserver --n 12 --seed 31
Use --ir for a frozen corpus; seed replay is only for a newly generated corpus.
"""
import argparse, hashlib, json, random, sys, time, traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from generate import MIXES, make_program, run_program                                             # noqa: E402
from teacher_trajectory_ir import convert as trajectory_ir, program_links       # noqa: E402
from natlang.checks import make_judge, run_checks                            # noqa: E402
from natlang.codebase import load_function                                   # noqa: E402
from natlang.decoder import LlamaServerDecoder                               # noqa: E402
from natlang.gen import codebases as C                                       # noqa: E402
from natlang.host import instantiate                                         # noqa: E402
from natlang.runtime import Runtime                                          # noqa: E402
from natlang.tool_agent import ToolAgent                                     # noqa: E402
from natlang.types import TypeEnv                                            # noqa: E402
from natlang.values import coerce, dump, dump_state                          # noqa: E402

WHERE = {"say": "shopkeeper/serve", "page_content": "webserver/handle", "submission_page": "webserver/handle"}   # the parent
NO_MARKUP_ATTACK = {"kind": "crisp", "code": "!/<\\s*script|<\\s*html|<\\s*body|javascript:/i.test(value)"}


def say_checks(args):
    action = args["action"]
    code, good, qty, price = (action[k] for k in ("code", "good", "qty", "price"))
    if code.startswith("sell"):
        question = (f"A sale of {qty} {good} at {price} coins each has happened. "
                    f"The total price is {qty * price} coins. Is the line consistent with that "
                    "completed sale? Accept natural wording and harmless personality. Reject a "
                    "different item, quantity, price, or a claim that the sale has not happened.")
    elif code == "quote":
        question = (f"The action is only a price quote for {good} at {price} coins, with no sale. "
                    f"Is the line consistent with that? Conversational shorthand such as "
                    f"'{good} is {price}' is a valid quote. Reject a different price or a claim "
                    "that goods have been handed over.")
    elif code == "counter_offer":
        question = (f"The action is a counter-offer for {qty} {good} at {price} coins each; "
                    "no sale has occurred. Is the line consistent with that offer? Accept natural "
                    "wording; reject a different price or a claim that goods were handed over.")
    elif code == "out_of_stock":
        question = (f"The action is out_of_stock for {good}; no sale occurs. Is the line "
                    "consistent with that unavailability, without claiming to hand over goods?")
    else:
        question = ("The action is ordinary chat; no sale occurs. Is the line consistent with "
                    "small talk, without claiming that goods were sold or handed over?")
    return [{"kind": "crisp", "code": "wordCount(value) >= 2 && wordCount(value) <= 45"},
            {"kind": "judge", "answer": True, "question": question}]


CHECKS = {
    "say": say_checks,
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
    """Collect missing cases, prioritizing keys that complete programs."""
    todo, seen = [], set()
    program_missing = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            oracles = record["semantics"].get("leaf_oracles", {})
            if not isinstance(oracles, dict):
                continue
            missing = set()
            for fn, oracle in oracles.items():
                if fn not in WHERE:
                    continue
                for case in oracle.get("cases", []):
                    if not case.get("template"):
                        continue
                    args = case["input"]
                    key = C.ref_key(fn, args)
                    if key not in C.REFERENCES:
                        missing.add(key)
                    if key not in seen and key not in C.REFERENCES:
                        seen.add(key)
                        todo.append((key, fn, args))
            if missing:
                program_missing.append(missing)

    sole = {key: sum(missing == {key} for missing in program_missing) for key, _, _ in todo}
    affected = {key: sum(key in missing for missing in program_missing) for key, _, _ in todo}
    # A key that is the last gap in a program gets first chance; ties favor keys
    # shared by more programs, then the stable hash order for reproducibility.
    return sorted(todo, key=lambda item: (-sole[item[0]], -affected[item[0]], item[0]))


def _leaf_audit_status(out, error):
    """Return auditable status/detail even when a leaf attempt raised."""
    if error is not None:
        return "exception", error
    return (out.kind, out.detail) if out is not None else ("exception", None)


def retry_keys(path: Path, status: str = "all") -> set[str]:
    """Select rejected keys from an immutable audit for a focused retry."""
    keys = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row["accepted"] and (status == "all" or row["status"] == status):
                keys.add(row["key"])
    return keys


def may_admit(function: str, dry_run: bool, admit_say: bool) -> bool:
    """Dialogue requires review while the Bonsai semantic judge is unreliable."""
    return not dry_run and (function != "say" or admit_say)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", type=Path, help="collect exact missing template cases from a frozen program IR")
    ap.add_argument("--families", nargs="*", default=["cb_shopkeeper", "cb_webserver"])
    ap.add_argument("--mix", default=None, help="a mix of scripts/generate.py, instead of --families")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--thinking", type=int, default=256)
    ap.add_argument("--turn-tokens", type=int,
                    help="optional maximum generated tokens for one teacher turn")
    ap.add_argument("--max-seconds", type=float, default=300,
                    help="wall-clock limit per teacher leaf (default: 300 seconds)")
    ap.add_argument("--retry-audit", type=Path,
                    help="retry only rejected keys from this audit")
    ap.add_argument("--retry-status", choices=("all", "quiesced", "done", "exception"), default="all")
    ap.add_argument("--limit", type=int, default=10**9, help="stop after this many leaves")
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="low")
    ap.add_argument("--system-file", type=Path, default=ROOT / "natlang/prompts/tools_teacher_compact.md")
    ap.add_argument("--dry-run", action="store_true", help="audit results without appending references")
    ap.add_argument("--admit-say", action="store_true",
                    help="allow automatic admission of dialogue that passes the Bonsai judge")
    ap.add_argument("--audit-out", type=Path, help="all accepted and rejected attempts, with checks and traces")
    ap.add_argument("--trajectory-out", type=Path,
                    help="linked teacher trajectory IR (defaults beside audit in --ir mode)")
    a = ap.parse_args()
    if a.turn_tokens is not None and a.turn_tokens < 1:
        ap.error("--turn-tokens must be positive")
    if a.max_seconds <= 0:
        ap.error("--max-seconds must be positive")
    if a.retry_status != "all" and not a.retry_audit:
        ap.error("--retry-status requires --retry-audit")
    audit_path = a.audit_out or ROOT / "runs" / f"teacher-leaves-audit-{time.time_ns()}.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists():
        ap.error(f"refusing overwrite: {audit_path}")
    trajectory_path = a.trajectory_out or (audit_path.with_suffix(".trajectory.ir.jsonl") if a.ir else None)
    if trajectory_path and not a.ir:
        ap.error("--trajectory-out requires --ir to link trajectories to frozen programs")
    if trajectory_path and trajectory_path.resolve() == audit_path.resolve():
        ap.error("trajectory IR and raw audit need different output paths")
    if trajectory_path and trajectory_path.exists():
        ap.error(f"refusing overwrite: {trajectory_path}")
    links = program_links(a.ir) if a.ir else None
    ir_hash = hashlib.sha256(a.ir.read_bytes()).hexdigest() if a.ir else None
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
    if a.retry_audit:
        wanted = retry_keys(a.retry_audit, a.retry_status)
        todo = [item for item in todo if item[0] in wanted]
    print(f"{len(todo)} generative leaves without a reference", flush=True)
    dec = LlamaServerDecoder(a.server, timeout=900, chat_extra={"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20,
                                         "chat_template_kwargs": {"reasoning_effort": a.reasoning_effort}},
                             tool_aliases={"call": "call_function"}, json_text_values=True)
    judge = make_judge(LlamaServerDecoder(a.server, timeout=300, chat_extra={"chat_template_kwargs": {"enable_thinking": False}}))
    C.REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    kept = admitted = 0
    for attempt_number, (k, fn, args) in enumerate(todo[: a.limit], 1):
        t0 = time.time()
        log, transcript, teacher_turns = [], [], []
        leaf_program, text, results, out, error = None, None, [], None, None
        ok = False
        try:
            definition = load_function(ROOT / "codebases" / (WHERE[fn] + ".nl")).codebase[fn]    # with the types it inherits
            root = instantiate(definition)
            env = root.env(TypeEnv())
            for name, value in args.items():
                root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
            leaf_program = dump_state(root)
            out, value = Runtime(lambda lam: ToolAgent(dec, temperature=a.temperature, system_prompt=prompt,
                                 validation_feedback="caller", log=log, transcript=transcript,
                                 teacher_turns=teacher_turns, turn_tokens=a.turn_tokens,
                                 max_seconds=a.max_seconds),
                                 max_episodes=4).run_root(root)
            text = dump(value) if out.kind == "done" else None
            results = run_checks(CHECKS[fn](args), text, judge) if text else []
            ok = bool(text) and all(r is True for _, r in results)
            if ok:
                if may_admit(fn, a.dry_run, a.admit_say):
                    with C.REF_FILE.open("a") as f:
                        f.write(json.dumps({"key": k, "function": fn, "args": args, "value": text}, ensure_ascii=False) + "\n")
                    admitted += 1
                kept += 1
        except Exception as exc:
            # A decoder, judge, harness, or serialization failure must not lose the
            # remaining frozen keys.  Keep the traceback in the audit for replay.
            error = {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc()}
            ok = False
        status, detail = _leaf_audit_status(out, error)
        audit_row = {"key":k,"function":fn,"args":args,"leaf_program":leaf_program,
                     "value":text,"accepted":ok,"admitted":ok and may_admit(fn, a.dry_run, a.admit_say),
                     "status":status,"detail":detail,"error":error,
                     "checks":results,"log":log,"transcript":transcript,
                     "teacher_turns":teacher_turns,"system_prompt":prompt,
                     "model_metadata":model_metadata,"temperature":a.temperature,
                     "thinking":a.thinking,"turn_tokens":a.turn_tokens,
                     "max_seconds":a.max_seconds,
                     "reasoning_effort":a.reasoning_effort,
                     "json_text_values":True}
        with audit_path.open("a") as f:
            f.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
        if trajectory_path:
            try:
                trajectory = trajectory_ir(audit_row, audit_path=audit_path,
                                           line_number=attempt_number, links=links,
                                           program_ir_hash=ir_hash)
                trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                with trajectory_path.open("a") as f:
                    f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            except Exception as exc:
                # The immutable audit can be converted again offline; a malformed
                # trajectory must not stop the remaining reference collection.
                errors_path = trajectory_path.with_suffix(trajectory_path.suffix + ".errors.jsonl")
                with errors_path.open("a") as f:
                    f.write(json.dumps({"audit_line": attempt_number, "key": k,
                                        "error": {"type": type(exc).__name__,
                                                  "message": str(exc),
                                                  "traceback": traceback.format_exc()}},
                                       ensure_ascii=False) + "\n")
        failed = [c.get("code") or c.get("question", "")[:60] for c, r in results if r is not True]
        shown = text or (out.kind if out is not None else f"exception: {error['type']}")
        label = "KEEP" if ok and audit_row["admitted"] else "review" if ok else "drop"
        print(f"[{fn}] {label} {time.time() - t0:4.0f}s  {shown[:90]!r}  {failed if failed else ''}", flush=True)
    print(f"passed checks {kept} of {min(len(todo), a.limit)}; admitted {admitted}; usage {dec.usage}; audit {audit_path}")


if __name__ == "__main__":
    main()
