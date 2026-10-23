"""Exact task collector and author interface for native program guidance search.

The author proposes natural-language program policy from development failures.
The collector and oracle score are crisp mechanisms. No weight adapter or flat
soft-block path is exposed by this module.
"""
from __future__ import annotations
import json
import hashlib
import base64
import re
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from ..eval.task_packet import read_jsonl
from .task_rollouts import build_queue, score
from ..common import memetic_teacher_ferry as teacher_ferry
ROOT = Path(__file__).resolve().parents[4]
GUIDANCE_WORDS = 120
PROPOSE_GUIDANCE = "You write guidance for a small language model that runs natural-language programs: it reads a call's\ninputs, uses tools (eval for exact computation, return_result to finish) and returns a typed result. The guidance is\nadded to its system prompt for every call of these tasks.\n\nAuthoring attempt context (stable across retries of this exact attempt):\n{attempt_context}\n\nCurrent guidance:\n{guidance}\n\nDevelopment task rows it ran and how they ended (outcome class, checks that failed, value returned, errors):\n{digest}\n\n{prior_rejection}\n\nUse only the development evidence above; no confirmation cases or outcomes are available to you. Write revised\nguidance that fixes these failures without naming tasks or their answers:\n1. Find the most common failure class above and what the model did wrong in it.\n2. State, as short numbered steps, what the model should do instead.\n3. Keep anything from current guidance that is not contradicted by failures.\n4. Use at most {words} words.\nReply with the guidance inside <guidance>...</guidance>."


def _header(response, name: str):
    getter = getattr(response, 'getheader', None)
    if getter:
        return getter(name)
    headers = getattr(response, 'headers', None)
    return headers.get(name) if headers is not None else None


def _author_custody(response, raw_request: bytes, raw_response: bytes) -> dict:
    request_sha = hashlib.sha256(raw_request).hexdigest()
    response_sha = hashlib.sha256(raw_response).hexdigest()
    request_id = _header(response, 'X-Natlang-Memetic-Request-Id')
    request_receipt_sha = _header(response, 'X-Natlang-Memetic-Request-Sha256')
    response_receipt_sha = _header(response, 'X-Natlang-Memetic-Response-Sha256')
    receipt_sha = _header(response, 'X-Natlang-Memetic-Ferry-Receipt-Sha256')
    receipt_encoded = _header(response, 'X-Natlang-Memetic-Ferry-Receipt')
    if (not isinstance(request_id, str) or not teacher_ferry.ID_RE.fullmatch(request_id)
            or request_receipt_sha != request_sha or response_receipt_sha != response_sha
            or not isinstance(receipt_encoded, str) or not isinstance(receipt_sha, str)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt_sha)):
        raise ValueError('teacher ferry response lacks exact request/response custody headers')
    try:
        receipt_raw = base64.urlsafe_b64decode(receipt_encoded + '=' * (-len(receipt_encoded) % 4))
        receipt = json.loads(receipt_raw)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError('teacher ferry receipt header is not valid base64 JSON') from error
    if (hashlib.sha256(receipt_raw).hexdigest() != receipt_sha
            or receipt.get('schema') != teacher_ferry.RECEIPT_SCHEMA
            or receipt.get('request_id') != request_id
            or receipt.get('request_sha256') != request_sha
            or receipt.get('response_sha256') != response_sha
            or receipt.get('response_bytes') != len(raw_response)
            or receipt.get('status') != 'complete'
            or not isinstance(receipt.get('hmac_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['hmac_sha256'])):
        raise ValueError('teacher ferry signed receipt does not bind the exact request and response bytes')
    teacher = receipt.get('teacher') or {}
    if (teacher.get('sandbox_id') != teacher_ferry.TEACHER_SANDBOX_ID
            or teacher.get('model_id') != teacher_ferry.TEACHER_MODEL_ID
            or teacher.get('serve_alias') != teacher_ferry.TEACHER_SERVE_ALIAS
            or teacher.get('revision') != teacher_ferry.TEACHER_REVISION
            or teacher.get('source_bindings') != list(teacher_ferry.TEACHER_SOURCE_BINDINGS)):
        raise ValueError('teacher ferry receipt is not linked to the pinned self-hosted teacher source')
    return {'schema': 'natlang.memetic-author-call-custody/1', 'request_id': request_id,
        'request_sha256': request_sha, 'response_sha256': response_sha,
        'ferry_receipt_sha256': receipt_sha, 'ferry_receipt': receipt}


def author_request_body(model: str, prompt: str, max_tokens: int = 4096) -> bytes:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def author_call(url: str, model: str, prompt: str, max_tokens: int = 4096) -> dict:
    endpoint = urlsplit(url)
    if model == teacher_ferry.TEACHER_SERVE_ALIAS and (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'):
        raise ValueError('the pinned self-hosted teacher must be reached through the durable evaluator-local ferry')
    raw = author_request_body(model, prompt, max_tokens)
    request_sha = hashlib.sha256(raw).hexdigest()
    headers = {"content-type": "application/json"}
    for attempt in range(5):
        try:
            request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=raw, headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=900) as response:
                raw_response = response.read()
                value = json.loads(raw_response)
                text = value["choices"][0]["message"].get("content") or ""
                if model == teacher_ferry.TEACHER_SERVE_ALIAS:
                    custody = _author_custody(response, raw, raw_response)
                else:
                    custody = {'schema': 'natlang.memetic-author-call-custody/1', 'request_id': None,
                        'request_sha256': request_sha, 'response_sha256': hashlib.sha256(raw_response).hexdigest(),
                        'ferry_receipt_sha256': None, 'ferry_receipt': None}
            if text.strip():
                return {'text': text, 'custody': custody}
        except Exception as error:  # retries resend the same exact body and retain one ferry idempotency key
            if attempt == 4:
                raise
            print(json.dumps({"event": "author_retry", "error": str(error)[:200]}), flush=True)
            time.sleep(30)
    return {'text': '', 'custody': None}


def propose_guidance(args, parent: dict, digest: str, *, attempt_context: dict,
                     prior_rejection: dict | None = None) -> dict:
    context_text = json.dumps(attempt_context, sort_keys=True, separators=(',', ':'))
    if prior_rejection:
        prior_text = ('A previous proposal was not adopted. Its development-only guidance and failure evidence are:\n'
            + 'Guidance: ' + prior_rejection['guidance'] + '\nDevelopment failures:\n'
            + prior_rejection['development_failures'])
    else:
        prior_text = 'There is no earlier rejected development proposal for this program.'
    authored = author_call(getattr(args, "author_ferry", None) or args.author, args.author_model, PROPOSE_GUIDANCE.format(
        attempt_context=context_text, guidance=parent["guidance"] or "(none)", digest=digest,
        prior_rejection=prior_text, words=GUIDANCE_WORDS))
    text = authored['text']
    match = re.search(r"<guidance>([\s\S]*?)</guidance>", text)
    words = (match.group(1) if match else text).split()
    if not words:
        raise ValueError("author returned no guidance")
    return {'text': " ".join(words[:GUIDANCE_WORDS]), 'author_call': authored['custody']}

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
