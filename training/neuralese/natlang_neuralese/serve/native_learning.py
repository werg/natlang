"""Native-state learning service operations.

This module keeps service policy at the HTTP boundary and uses the same native
Engine cache/step functions as generation. Native-state scoring never routes
through legacy PortHeads or converts an imported state to a flat vector.
"""

from __future__ import annotations

import torch

from .chat import RequestError
from .engine import GenerationRequest, Sequence
from .grad import GradSession


def _native_plan(engine, items):
    """Translate the shared renderer's prompt items into native cache segments."""
    plan = []
    run = []
    for kind, value in items:
        if kind == "tok":
            run.append(int(value))
            continue
        if kind != "block" or not isinstance(value, str) or not value.startswith("nzs1_"):
            raise RequestError("neuralese-native-state-unavailable",
                               "native decisions require exact native-state imports")
        if run:
            plan.append(("tokens", run))
            run = []
        plan.append(("native_state", value))
    if run:
        plan.append(("tokens", run))
    return plan


def _decision_render(engine, messages, tools, options):
    if not isinstance(options, list) or not options:
        raise RequestError("neuralese-decision", "a decision needs at least one option")
    session = GradSession(engine)
    prompt = None
    continuations = []
    for option in options:
        if not isinstance(option, str):
            raise RequestError("neuralese-decision", "options are reply texts")
        before, rest = session._target_items(messages, tools, {"role": "assistant", "content": option})
        if prompt is not None and before != prompt:
            raise RequestError("neuralese-decision", "options rendered different prompts")
        if any(kind != "tok" for kind, _ in rest):
            raise RequestError("neuralese-decision", "options must be ordinary assistant text")
        prompt = before
        continuations.append([int(value) for _, value in rest])
    return _native_plan(engine, prompt), continuations


def _score_from_cache(engine, cache, logits, continuation):
    score = torch.zeros((), dtype=torch.float32, device=logits.device)
    with torch.no_grad():
        for token in continuation:
            score = score + torch.log_softmax(logits.float(), dim=-1)[0, token]
            output = engine._native_forward([token], cache)
            cache, logits = output["cache"], engine._native_logits(output["h_final"])
    return float(score), len(continuation)


def _score_item(engine, item, adapters=None):
    if not isinstance(item, dict):
        raise RequestError("neuralese-decision", "each decision item must be an object")
    options = item.get("continuations", item.get("options"))
    plan, continuations = _decision_render(engine, item.get("messages") or [], item.get("tools"), options)
    bindings = item.get('adapters', adapters)
    return engine._native_run(lambda: _score_rendered(engine, item, plan, continuations, bindings), bindings)


def _score_rendered(engine, item, plan, continuations, bindings):
    request = GenerationRequest(messages=item.get("messages") or [], tools=item.get("tools"), max_tokens=0,
                                adapters=bindings)
    sequence = Sequence(request, None, prompt_plan=plan)
    engine._native_prefill(sequence)
    base_cache, base_logits = sequence.cache, sequence.logits
    results = [_score_from_cache(engine, base_cache, base_logits, own) for own in continuations]
    return {"log_probs": [score for score, _ in results], "tokens": [count for _, count in results]}


def decide_native(engine, body: dict) -> dict:
    """Score candidate replies from a single exact native prompt cache."""
    return _score_item(engine, {"messages": body.get("messages") or [], "tools": body.get("tools"),
                                "options": body.get("options") or []}, body.get('adapters'))


def decide_many_native(engine, body: dict) -> dict:
    """Batch decisions without changing per-item semantics; failures remain item-local."""
    items = body.get("items")
    if not isinstance(items, list):
        raise RequestError("neuralese-decision", "items must be a list")
    results = []
    for item in items:
        try:
            results.append(_score_item(engine, item, body.get("adapters")))
        except Exception as error:  # item failures are part of the shared score contract
            code = error.code if isinstance(error, RequestError) else "internal"
            results.append({"error": f"{code}: {error}"[:500]})
    return {"results": results}
