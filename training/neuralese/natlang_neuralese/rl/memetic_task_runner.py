"""Exact task collector and author interface for native program guidance search.

The author proposes natural-language program policy from development failures.
The collector and oracle score are crisp mechanisms. No weight adapter or flat
soft-block path is exposed by this module.
"""
from __future__ import annotations
import json
import hashlib
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from ..eval.task_packet import read_jsonl
from .task_rollouts import build_queue, score
ROOT = Path(__file__).resolve().parents[4]
GUIDANCE_WORDS = 120
PROPOSE_GUIDANCE = "You write guidance for a small language model that runs natural-language programs: it reads a call's\ninputs, uses tools (eval for exact computation, return_result to finish) and returns a typed result. The guidance is\nadded to its system prompt for every call of these tasks.\n\nCurrent guidance:\n{guidance}\n\nTask rows it ran and how they ended (outcome class, the checks that failed, the value it returned, errors):\n{digest}\n\nWrite revised guidance that fixes the failures without naming these tasks or their answers:\n1. Find the most common failure class above and what the model did wrong in it.\n2. State, as short numbered steps, what the model should do instead.\n3. Keep anything from the current guidance that is not contradicted by the failures.\n4. Use at most {words} words.\nReply with the guidance inside <guidance>...</guidance>."


def author_call(url: str, model: str, prompt: str, max_tokens: int = 4096) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(5):
        try:
            request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"content-type": "application/json"})
            text = json.load(urllib.request.urlopen(request, timeout=900))["choices"][0]["message"].get("content") or ""
            if text.strip():
                return text
        except Exception as error:  # the author server can be restarting; the caller records a failed operator
            if attempt == 4:
                raise
            print(json.dumps({"event": "author_retry", "error": str(error)[:200]}), flush=True)
            time.sleep(30)
    return ""

def propose_guidance(args, parent: dict, digest: str) -> str:
    text = author_call(args.author, args.author_model, PROPOSE_GUIDANCE.format(
        guidance=parent["guidance"] or "(none)", digest=digest, words=GUIDANCE_WORDS))
    match = re.search(r"<guidance>([\s\S]*?)</guidance>", text)
    words = (match.group(1) if match else text).split()
    if not words:
        raise ValueError("author returned no guidance")
    return " ".join(words[:GUIDANCE_WORDS])

def default_system_prompt() -> str:
    script = ("import(process.argv[1]).then(m => process.stdout.write(m.defaultSystemPrompt))")
    return subprocess.run(["node", "-e", script, str(ROOT / "ts-host/dist/teacher/collector.js")], check=True,
                          capture_output=True, text=True).stdout

def guided_system(base: str, guidance: str) -> str:
    """The runtime's program-guidance wrapper (ts-host adaptation/prompts.ts programGuidance)."""
    return base + ("\n\n<natlang_program_guidance>\n" + guidance + "\n</natlang_program_guidance>\n" if guidance else "")

def evaluate(args, ind: dict, packet: Path, base_system: str) -> dict:
    """Run the individual on every packet row (`samples` each, the same seeds for every individual) and verify."""
    out = Path(args.out) / "evals" / ind["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "packet.jsonl").write_text(packet.read_text())
    # The shared rollout scorer needs the exact packet binding to authenticate
    # complete budget-terminated execution prefixes without a terminal result row.
    from ..common.jsonio import write_json_durable
    write_json_durable(out / 'preparation.json', {
        'schema': 'natlang.memetic-task-capture-preparation/1', 'root_mode': 'authored',
        'packet_sha256': hashlib.sha256((out / 'packet.jsonl').read_bytes()).hexdigest(),
        'policy_checkpoint_sha256': args.policy_checkpoint_sha256})
    system = out / "system.md"
    system.write_text(guided_system(base_system, ind["guidance"]))
    capture_config = out / 'native-capture.json'
    capture_config.write_text(json.dumps({'x_natlang_rl_capture': True}))
    entries = build_queue(out / "packet.jsonl", args.rows, args.samples, out, temperature=args.temperature,
                          case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                          max_turns=args.max_turns, seed_base=args.seed_base)
    for entry in entries:
        entry["system_file"] = str(system)
    (out / "queue.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    started = time.time()
    procs = []
    for s in range(args.shards):
        (shards / f"queue-{s}.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries[s::args.shards]))
        log = open(shards / f"queue-{s}.log", "w")
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/run_bonsai_queue.py"), str(shards / f"queue-{s}.jsonl"),
                                       str(shards / f"journal-{s}.jsonl"), "--runtime", str(ROOT / "ts-host"),
                                       "--server", args.server, "--chat-completions-url", args.server.rstrip("/") + "/v1/chat/completions",
                                       "--model-id", args.model_id, "--model-concurrency", "2",
                                       "--neuralese-server", "--chat-request-config", str(capture_config)], stdout=log, stderr=subprocess.STDOUT))
    for proc in procs:
        proc.wait()
    (out / "journal.jsonl").write_text("".join((shards / f"journal-{s}.jsonl").read_text()
                                               for s in range(args.shards) if (shards / f"journal-{s}.jsonl").exists()))
    report = score(out, out / "rollouts.jsonl")
    rewards = read_jsonl(out / "rewards.jsonl")
    scored = [r for r in rewards if r["reward"] is not None]
    fit = {"fitness": sum(r["reward"] for r in scored) / len(scored) if scored else 0.0,
           "successes": report["successes"], "scored": report["scored"], "rollouts": report["rollouts"],
           "budget_stops": report["outcome_counts"].get("budget_exhaustion", 0),
           "outcomes": report["outcome_counts"], "groups_with_spread": report["groups_with_spread"],
           "per_row": [sum(r["reward"] == 1.0 for r in rewards if r["group"] == g) for g in range(args.rows)],
           "seconds": round(time.time() - started, 1), "dir": str(out)}
    return fit

def failure_digest(fit: dict, limit: int = 6) -> str:
    """Failed rollouts of an evaluation: outcome class, failed checks, returned value, errors. Never the gold."""
    out, items = Path(fit["dir"]), []
    for reward in read_jsonl(out / "rewards.jsonl"):
        if reward["reward"] != 0.0 or len(items) >= limit:
            continue
        path = out / f"row-{reward['group']:03d}" / f"sample-{reward['sample']:02d}" / "results.jsonl"
        result = (read_jsonl(path) or [{}])[0] if path.exists() else {}
        outcome = result.get("outcome") or {}
        failed = [k for k, v in (outcome.get("checks") or {}).items() if v is False]
        errors = [str(f.get("message", ""))[:200] for f in outcome.get("scope_failures") or [] if isinstance(f, dict)]
        diagnostics = [str(d)[:200] for a in outcome.get("action_ledger") or [] if isinstance(a, dict)
                       for d in (a.get("diagnostics") or [])][-3:]
        items.append(f"- task family {reward['family']}: {reward['outcome']}; failed checks {failed or 'none'}; "
                     f"returned {json.dumps(outcome.get('value'), ensure_ascii=False)[:300]}; errors {errors[:2] + diagnostics}")
    return "\n".join(items) or "(no failures)"
