"""Author interface and guidance prompt wrapper for native program guidance search.

The author proposes natural-language program policy from development failures.
Evaluation runs through the production collector (rl.production.collect, called by
memetic_program.evaluate); the oracle score is the shared task scorer. No weight
adapter or flat soft-block path is exposed by this module.
"""
from __future__ import annotations
import json
import hashlib
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from ..eval.task_packet import read_jsonl
ROOT = Path(__file__).resolve().parents[4]
GUIDANCE_WORDS = 120
PROPOSE_GUIDANCE = "You write guidance for a small language model that runs natural-language programs: it reads a call's\ninputs, uses tools (eval for exact computation, return_result to finish) and returns a typed result. The guidance is\nadded to its system prompt for every call of these tasks.\n\nAuthoring attempt context (stable across retries of this exact attempt):\n{attempt_context}\n\nCurrent guidance:\n{guidance}\n\nDevelopment task rows it ran and how they ended (outcome class, checks that failed, value returned, errors):\n{digest}\n\n{prior_rejection}\n\nUse only the development evidence above; no confirmation cases or outcomes are available to you. Write revised\nguidance that fixes these failures without naming tasks or their answers:\n1. Find the most common failure class above and what the model did wrong in it.\n2. State, as short numbered steps, what the model should do instead.\n3. Keep anything from current guidance that is not contradicted by failures.\n4. Use at most {words} words.\nReply with the guidance inside <guidance>...</guidance>."


def author_request_body(model: str, prompt: str, max_tokens: int = 4096) -> bytes:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def author_call(url: str, model: str, prompt: str, max_tokens: int = 4096) -> dict:
    """One chat call to the served native author, with exact request/response byte custody."""
    raw = author_request_body(model, prompt, max_tokens)
    request_sha = hashlib.sha256(raw).hexdigest()
    headers = {"content-type": "application/json"}
    for attempt in range(5):
        try:
            request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=raw, headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=900) as response:
                raw_response = response.read()
            text = json.loads(raw_response)["choices"][0]["message"].get("content") or ""
            if text.strip():
                return {'text': text, 'custody': {'schema': 'natlang.memetic-author-call-custody/2',
                        'request_sha256': request_sha, 'response_sha256': hashlib.sha256(raw_response).hexdigest()}}
        except Exception as error:  # retries resend the same exact body
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
    authored = author_call(args.author, args.author_model, PROPOSE_GUIDANCE.format(
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
