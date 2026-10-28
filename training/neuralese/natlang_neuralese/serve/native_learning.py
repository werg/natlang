"""Native-state learning service operations.

This module keeps service policy at the HTTP boundary and uses the same native
Engine cache/step functions as generation. Native-state scoring never routes
through legacy PortHeads or converts an imported state to a flat vector.
"""

from __future__ import annotations

import torch

from ..common import stable_softmax
from .chat import RequestError
from .engine import GenerationRequest, Sequence
from .targets import target_items


def _native_plan(engine, items):
    """Translate the shared renderer's prompt items into native cache segments."""
    plan = []
    run = []
    for kind, value in items:
        if kind == "tok":
            run.append(int(value))
            continue
        if kind not in ("block","block_stop") or not isinstance(value.get("id") if isinstance(value,dict) else value, str) or not (value["id"] if isinstance(value,dict) else value).startswith("nzs1_"):
            raise RequestError("neuralese-native-state-unavailable",
                               "native decisions require exact native-state imports")
        if run:
            plan.append(("tokens", run))
            run = []
        plan.append(("native_state_constant" if kind=='block_stop' else "native_state", value))
    if run:
        plan.append(("tokens", run))
    return plan


def decision_items(renderer, messages, tools, options):
    """The decision prompt items and each option's complete closed continuation token IDs (end of message
    included). ``renderer`` is a serving engine or ``data.decision_graph.tokenizer_renderer`` (CPU conversion):
    one rendering for serving, training rows and recorded captures."""
    if not isinstance(options, list) or not options:
        raise RequestError("neuralese-decision", "a decision needs at least one option")
    prompt = None
    continuations = []
    for option in options:
        if not isinstance(option, str):
            raise RequestError("neuralese-decision", "options are reply texts")
        try:
            before, rest = target_items(renderer, messages, tools, {"role": "assistant", "content": option})
        except RequestError as error:
            raise RequestError("neuralese-decision", str(error).removeprefix(f"{error.code}: ")) from error
        if prompt is not None and before != prompt:
            raise RequestError("neuralese-decision", "options rendered different prompts")
        if any(kind != "tok" for kind, _ in rest):
            raise RequestError("neuralese-decision", "options must be ordinary assistant text")
        prompt = before
        continuations.append([int(value) for _, value in rest])
    return prompt, continuations


def _decision_render(engine, messages, tools, options):
    prompt, continuations = decision_items(engine, messages, tools, options)
    return _native_plan(engine, prompt), continuations


def _score_from_cache(engine, cache, logits, continuation):
    score = torch.zeros((), dtype=torch.float32, device=logits.device)
    with torch.no_grad():
        for token in continuation:
            score = score + stable_softmax.token_logprobs(logits[:1], torch.tensor([token], device=logits.device))[0]
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
    privileged = item.get('x_natlang_privileged_context')
    if privileged is not None and not item.get('native_recording'):
        raise RequestError('bad-request', 'x_natlang_privileged_context renders a captured JOLT+ teacher decision; '
                           'request native_recording')
    request = GenerationRequest(messages=item.get("messages") or [], tools=item.get("tools"), max_tokens=0,
                                adapters=bindings, native_state_modes=item.get('native_state_modes'),
                                rl_capture=bool(item.get('native_recording')),native_recording=bool(item.get('native_recording')),
                                privileged_context=privileged)
    sequence = Sequence(request, None, prompt_plan=plan)
    engine._native_prefill(sequence)
    base_cache, base_logits = sequence.cache, sequence.logits
    results = [_score_from_cache(engine, base_cache, base_logits, own) for own in continuations]
    response={"log_probs": [score for score, _ in results], "tokens": [count for _, count in results],
              "native_state_imports": sequence.native_imports}
    if item.get('native_recording'):
        response['native_decision_capture']={'schema':'natlang.native-decision-capture/1',
            'policy':dict(sequence.native_rl_policy),'adapter_binding':sequence.native_rl_adapter_binding,
            **({'privileged': sequence.native_guidance_receipt} if sequence.native_guidance_receipt is not None else {}),
            'prompt':sequence.native_rl_prompt,'continuations':continuations,
            'native_state_imports':sequence.native_imports}
    return response


def decide_native(engine, body: dict) -> dict:
    """Score candidate replies from a single exact native prompt cache."""
    return _score_item(engine, {"messages": body.get("messages") or [], "tools": body.get("tools"),
                                "options": body.get("options") or [], "native_state_modes": body.get("native_state_modes"),
                                "native_recording":body.get("native_recording",False),
                                **({'x_natlang_privileged_context': body['x_natlang_privileged_context']}
                                   if body.get('x_natlang_privileged_context') is not None else {})}, body.get('adapters'))


def decide_many_native(engine, body: dict) -> dict:
    """Batch decisions without changing per-item semantics; failures remain item-local."""
    items = body.get("items")
    if not isinstance(items, list):
        raise RequestError("neuralese-decision", "items must be a list")
    results = []
    for item in items:
        try:
            results.append(_score_item(engine, {"native_state_modes": body.get("native_state_modes"), "native_recording":body.get("native_recording",False), **item}, body.get("adapters")))
        except Exception as error:  # item failures are part of the shared score contract
            # A RequestError's text already starts with its code.
            message = str(error) if isinstance(error, RequestError) else f"internal: {error}"
            results.append({"error": message[:500]})
    return {"results": results}
