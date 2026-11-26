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


# Every field natlang's client puts on a /v1/natlang/score item (ts-host/src/model/decision-scoring.ts batchScoreItem),
# and what this server (native engine and SGLang actor) does with it. A field the llama.cpp fork serves and this server
# does not is a parity gap recorded in docs/ENGINE_PARITY.md; tests/neuralese/test_engine_parity.py fails when the client
# sends a field that is not listed here.
SCORE_ITEM_FIELDS = {
    'messages': 'served',
    'continuations': 'served',
    'adapters': 'served',
    'native_state_modes': 'served',
    'native_recording': 'served',
    'cache_messages': 'cache hint: the native engine ignores it (its prefix cache reuses grid blocks of any prompt); the '
                      'SGLang actor prefills the opening once before scoring, so its radix cache stores the conv state '
                      'at the boundary (fork 33230160b, natlang_track_input_end) or within 255 tokens of it (older forks) '
                      'for every input (sglang_actor._sglang_warm_opening; '
                      'NATLANG_SGLANG_WARM_OPENINGS=0 opts out; ENGINE_PARITY.md)',
    'open_reply': 'served: each continuation is the start of the assistant reply, scored without its end of message '
                  '(decision_items open_reply; the fork: continue_final_message "content"); results and the decision '
                  'capture echo open_reply: true',
    'reasoning': 'served: the reply\'s reasoning (a string), sampled before the options are scored; each option is the '
                 'content after it, rendered as the template renders an assistant message with that reasoning '
                 '(reasoning_content; LFM3 <think>R</think>option), so the reasoning and the tokens every option shares '
                 'are prompt (decision_items reasoning); results echo reasoning: true and the decision capture records '
                 'the text (its prompt ids hold its tokens)',
    'split_tokenization': 'served: each option also scores its split form (the options\' common text prefix, its own '
                          'middle and the common suffix, each tokenized on its own) when that differs from the '
                          'canonical tokens, and its score is the logsumexp of both (decision_sequences); results echo '
                          'split_tokenization: true and tokenizations per option, the decision capture adds '
                          'continuation_alternatives',
    'reasoning_ids': 'served: with reasoning, its generation\'s sampled ids replace the re-encoded reasoning tokens, '
                     'kept up to the first id of the token that follows the reasoning (LFM3 </think>); results echo '
                     'reasoning_ids: true and the capture records reasoning_ids_kept (decision_sequences)',
    'x_natlang_privileged_context': 'served: captured JOLT+ teacher decision guidance context (native_learning.decision_items)',
}


def open_reply_flag(item):
    """An item's ``open_reply`` (absent: False); anything but a boolean is refused."""
    value = item.get('open_reply', False)
    if type(value) is not bool:
        raise RequestError("neuralese-decision", "open_reply must be a boolean")
    return value


def reasoning_text(item):
    """An item's ``reasoning`` (absent: None, a readout without reasoning); anything but a string is refused."""
    value = item.get('reasoning')
    if value is not None and not isinstance(value, str):
        raise RequestError("neuralese-decision", "reasoning must be a string")
    return value


def split_tokenization_flag(item):
    """An item's ``split_tokenization`` (absent: False); anything but a boolean is refused."""
    value = item.get('split_tokenization', False)
    if type(value) is not bool:
        raise RequestError("neuralese-decision", "split_tokenization must be a boolean")
    return value


def _word(ch):
    """A word character of ``split_pieces``: ASCII [A-Za-z0-9_] or any code point >= 0x80 (the fork and the client
    classify identically, without Unicode tables)."""
    return ch == '_' or 'a' <= ch <= 'z' or 'A' <= ch <= 'Z' or '0' <= ch <= '9' or ord(ch) >= 0x80


def split_pieces(options):
    """Per option its split form's text pieces [P, M_i, S] (empty pieces dropped): P the options' longest common
    character prefix, S the longest common character suffix of what remains after P, each cut back to a word boundary
    (P never ends, and S never starts, between two word characters of an option: ``"spam"``/``"ham"`` share ``am"``
    but split as ``"``, ``spam``/``ham``, ``"``; ``_word``). P is cut first, S is computed on what remains. One option: None."""
    if len(options) < 2:
        return None
    import os.path
    p = len(os.path.commonprefix(list(options)))
    while p and any(len(o) > p and _word(o[p - 1]) and _word(o[p]) for o in options):
        p -= 1
    rests = [option[p:] for option in options]
    n = len(os.path.commonprefix([rest[::-1] for rest in rests]))
    while n and any(len(r) > n and _word(r[-n]) and _word(r[-n - 1]) for r in rests):
        n -= 1
    return [[piece for piece in (option[:p], rest[:len(rest) - n], rest[len(rest) - n:]) if piece]
            for option, rest in zip(options, rests)]


def _content_tokens(renderer, text):
    """``text`` tokenized as message content: no special-token parsing (the template renders content as text)."""
    return list(renderer.tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"])


def reasoning_ids_of(item):
    """An item's ``reasoning_ids`` (absent: None): a list of nonnegative ints, only beside ``reasoning``."""
    value = item.get('reasoning_ids')
    if value is None:
        return None
    if not isinstance(item.get('reasoning'), str):
        raise RequestError("neuralese-decision", "reasoning_ids come with the reasoning they were sampled as")
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise RequestError("neuralese-decision", "reasoning_ids must be a list of token ids")
    return value


def decision_items(renderer, messages, tools, options, open_reply=False, reasoning=None, reasoning_ids=None):
    """``decision_sequences`` without split tokenization: (prompt items, canonical continuation ids per option)."""
    prompt, continuations, _ = decision_sequences(renderer, messages, tools, options, open_reply, reasoning,
                                                  reasoning_ids=reasoning_ids)
    return prompt, continuations


def _splice_reasoning(renderer, ids, reasoning, reasoning_ids, info):
    """``ids`` (a reply's rendered tokens: R's tokens, then the text after R) with R's re-encoded tokens replaced by the
    sampled ``reasoning_ids`` up to (not including) the first id of the token that follows R (LFM3 ``</think>``)."""
    if not reasoning:
        raise RequestError("neuralese-decision", "reasoning_ids: the rendered reply does not hold the reasoning text")
    encoded = _content_tokens(renderer, reasoning)
    if ids[:len(encoded)] != encoded or len(ids) <= len(encoded):
        raise RequestError("neuralese-decision", "reasoning_ids: the rendered reply does not hold the reasoning text")
    vocab = len(renderer.tokenizer)
    if any(token >= vocab for token in reasoning_ids):
        raise RequestError("neuralese-decision", f"reasoning_ids must be below the vocabulary size {vocab}")
    follow = ids[len(encoded)]
    kept = reasoning_ids[:reasoning_ids.index(follow)] if follow in reasoning_ids else list(reasoning_ids)
    info['reasoning_ids_kept'] = len(kept)
    return list(kept) + ids[len(encoded):]


def decision_sequences(renderer, messages, tools, options, open_reply=False, reasoning=None, split_tokenization=False,
                       reasoning_ids=None, info=None):
    """The decision prompt items and each option's own closed continuation token IDs (end of message included; the
    tokens every option shares are part of the prompt, ``_shared_into_prompt``). ``renderer`` is a serving engine or ``data.decision_graph.tokenizer_renderer`` (CPU conversion):
    one rendering for serving, training rows and recorded captures.

    ``open_reply``: each option is the start of the reply (a field-mode decision step), scored without the end of
    message: the template's rendering of the final assistant message with that content, cut at the end of the content
    (HF ``continue_final_message``, the fork's ``continue_final_message: "content"``). Chat templates end a message with
    a special end token that the text before it never merges with, so that is the closed continuation without its
    final end-of-message token (for LFM3: ``</think>`` + content; tests/neuralese/test_decision_open_reply.py checks it
    against the template's own continue_final_message rendering).

    ``reasoning``: the reply's reasoning, sampled before the options are scored (a reasoned readout, the default of
    ``readout: decision``). Each option is the content of an assistant message with that reasoning
    (``reasoning_content``), so every option continues ``<think>R</think>`` on LFM3; those tokens are shared and so
    move to the prompt. None (the instant readout) renders the reply without reasoning (LFM3: ``<think></think>``).

    ``split_tokenization``: the model may write an option in another token split than the canonical one (``"sp``,
    ``am"`` vs ``"``, ``spam``, ``"``). Each option's split form is its content as the pieces of ``split_pieces``, each
    tokenized on its own, between exactly the canonical form's tokens before the content (generation prefix,
    ``</think>``) and after it (``<|im_end|>`` when closed). Returns (prompt items, canonical continuation ids,
    alternatives: per option the list of its other id sequences, empty when the split equals the canonical form); the
    option's score is the logsumexp over its sequences (``option_log_scores``). The shared prefix moved to the prompt is
    the one of all sequences.

    ``reasoning_ids`` (with ``reasoning``): the reasoning generation's sampled ids. The engines re-encode the reasoning
    text when they render it, which can leave the ids the model sampled (``Hel``, ``lo`` vs ``Hello``); the sampled ids
    replace the re-encoded ones, kept up to the first id of the token that follows the reasoning (``</think>``)
    (``_splice_reasoning``; ``info['reasoning_ids_kept']`` their count). They are shared, so they are prompt."""
    if not isinstance(options, list) or not options:
        raise RequestError("neuralese-decision", "a decision needs at least one option")
    prompt = None
    continuations = []
    for option in options:
        if not isinstance(option, str):
            raise RequestError("neuralese-decision", "options are reply texts")
        try:
            before, rest = target_items(renderer, messages, tools, {"role": "assistant", "content": option,
                                                                    **({"reasoning_content": reasoning} if reasoning is not None else {})})
        except RequestError as error:
            raise RequestError("neuralese-decision", str(error).removeprefix(f"{error.code}: ")) from error
        if prompt is not None and before != prompt:
            raise RequestError("neuralese-decision", "options rendered different prompts")
        if any(kind != "tok" for kind, _ in rest):
            raise RequestError("neuralese-decision", "options must be ordinary assistant text")
        prompt = before
        ids = [int(value) for _, value in rest]
        if reasoning_ids is not None:
            ids = _splice_reasoning(renderer, ids, reasoning, reasoning_ids, info if info is not None else {})
        if open_reply:
            end = renderer.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if not ids or ids[-1] != end:
                raise RequestError("neuralese-decision", "an open reply needs a template that ends a message with <|im_end|>")
            ids = ids[:-1]
            if not ids:
                raise RequestError("neuralese-decision", "an open reply option needs reply text")
        continuations.append(ids)
    alternatives = [[] for _ in continuations]
    pieces = split_pieces(options) if split_tokenization else None
    if pieces is not None:
        end_id = renderer.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for index, (option, ids) in enumerate(zip(options, continuations)):
            if not open_reply and (not ids or ids[-1] != end_id):
                continue   # a template whose message end is not <|im_end|>: the tokens after the content are unknown
            post = [] if open_reply else [end_id]
            content = _content_tokens(renderer, option)
            end = len(ids) - len(post)
            # the canonical content must be the content tokenized alone, so the tokens around it are known exactly
            if end < len(content) or ids[end - len(content):end] != content or ids[end:] != post:
                continue
            split = ids[:end - len(content)] + [t for piece in pieces[index] for t in _content_tokens(renderer, piece)] + post
            if split != ids:
                alternatives[index].append(split)
    return _shared_into_prompt(prompt, continuations, alternatives)


def _shared_into_prompt(prompt, continuations, alternatives=None):
    """Tokens every option begins with (LFM3's ``</think>`` closing the empty think block, a field step's chosen
    fields, the record's opening ``{"``) are context, not part of a choice: they move to the prompt, and each option
    scores only its own tokens (at least its last one). They add the same log-probability to every option, which the
    readout's softmax cancels only when every option's request computes them identically; scored once in the prompt
    they cancel exactly (an SGLang score sends one request per option, and the -16 nats of ``</think>`` differed between
    them by up to ~0.6 nats, decision-parity hunt 2026-11-23). Training rows (sft_sources, decision captures, tune rows)
    render through here and stay identical. NATLANG_DECISION_SHARED_PREFIX=option keeps them in every option.
    ``alternatives`` (per option its other id sequences): the prefix is common to every sequence of every option.
    Returns (prompt, continuations, alternatives)."""
    import os
    alternatives = alternatives if alternatives is not None else [[] for _ in continuations]
    sequences = list(continuations) + [ids for alts in alternatives for ids in alts]
    if len(continuations) < 2 or os.environ.get("NATLANG_DECISION_SHARED_PREFIX", "prompt") == "option":
        return prompt, continuations, alternatives
    first, limit = sequences[0], min(len(ids) for ids in sequences) - 1
    shared = 0
    while shared < limit and all(ids[shared] == first[shared] for ids in sequences):
        shared += 1
    if not shared:
        return prompt, continuations, alternatives
    return (list(prompt) + [("tok", int(token)) for token in first[:shared]], [ids[shared:] for ids in continuations],
            [[ids[shared:] for ids in alts] for alts in alternatives])


def option_log_scores(sequence_scores, continuations, alternatives):
    """Per option the logsumexp of its sequences' scores: ``sequence_scores`` holds the canonical continuations' scores,
    then every alternative's in ``alternatives`` order (an option without alternatives keeps its score exactly)."""
    import math
    scores = list(sequence_scores[:len(continuations)])
    at = len(continuations)
    for index, alts in enumerate(alternatives or ()):
        for _ in alts:
            a, b = scores[index], sequence_scores[at]
            high = max(a, b)
            scores[index] = high + math.log(math.exp(a - high) + math.exp(b - high))
            at += 1
    return scores


def _decision_render(engine, messages, tools, options, open_reply=False, reasoning=None, split_tokenization=False,
                     reasoning_ids=None, info=None):
    """(native plan, canonical continuations, alternatives per option)."""
    if split_tokenization or reasoning_ids is not None:
        prompt, continuations, alternatives = decision_sequences(engine, messages, tools, options, open_reply, reasoning,
                                                                 split_tokenization, reasoning_ids, info)
        alternatives = alternatives if split_tokenization else None
    else:
        prompt, continuations = decision_items(engine, messages, tools, options, open_reply, reasoning)
        alternatives = None
    return _native_plan(engine, prompt), continuations, alternatives


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
    info = {}
    plan, continuations, alternatives = _decision_render(engine, item.get("messages") or [], item.get("tools"), options,
                                                         open_reply_flag(item), reasoning_text(item),
                                                         split_tokenization_flag(item), reasoning_ids_of(item), info)
    if 'reasoning_ids_kept' in info:
        item = {**item, '_reasoning_ids_kept': info['reasoning_ids_kept']}
    bindings = item.get('adapters', adapters)
    extra = {'alternatives': alternatives} if alternatives is not None else {}
    serve = getattr(engine, 'sglang_score', None)  # the SGLang actor front scores on SGLang when it can
    if serve is not None:
        result = serve(item, plan, continuations, bindings, **extra)
        if result is not None:
            return result
    return engine._native_run(lambda: _score_rendered(engine, item, plan, continuations, bindings, **extra), bindings)


def decision_request(item, bindings):
    privileged = item.get('x_natlang_privileged_context')
    if privileged is not None and not item.get('native_recording'):
        raise RequestError('bad-request', 'x_natlang_privileged_context renders a captured JOLT+ teacher decision; '
                           'request native_recording')
    return GenerationRequest(messages=item.get("messages") or [], tools=item.get("tools"), max_tokens=0,
                             adapters=bindings, native_state_modes=item.get('native_state_modes'),
                             rl_capture=bool(item.get('native_recording')),native_recording=bool(item.get('native_recording')),
                             privileged_context=privileged)


def decision_sequence_list(continuations, alternatives):
    """Every id sequence an item scores: the canonical continuations, then each option's alternatives in order."""
    return list(continuations) + [ids for alts in alternatives or () for ids in alts]


def decision_response(item, sequence, continuations, results, alternatives=None):
    """``results``: (score, tokens) per ``decision_sequence_list`` entry; an option's log_prob is the logsumexp of its
    sequences' scores, its tokens the canonical count."""
    scores = option_log_scores([score for score, _ in results], continuations, alternatives)
    response={"log_probs": scores, "tokens": [count for _, count in results[:len(continuations)]],
              "native_state_imports": sequence.native_imports}
    spliced = item.get('_reasoning_ids_kept')
    if spliced is not None:
        response['reasoning_ids'] = True
    split = alternatives is not None
    if split:
        response['split_tokenization'] = True
        response['tokenizations'] = [1 + len(alts) for alts in alternatives]
    opened = {'open_reply': True} if item.get('open_reply') is True else {}
    response.update(opened)
    reasoning = item.get('reasoning') if isinstance(item.get('reasoning'), str) else None
    if reasoning is not None:
        response['reasoning'] = True
    if item.get('native_recording'):
        response['native_decision_capture']={'schema':'natlang.native-decision-capture/1', **opened,
            **({'reasoning': reasoning} if reasoning is not None else {}),
            **({'reasoning_ids_kept': spliced} if spliced is not None else {}),
            'policy':dict(sequence.native_rl_policy),'adapter_binding':sequence.native_rl_adapter_binding,
            **({'privileged': sequence.native_guidance_receipt} if sequence.native_guidance_receipt is not None else {}),
            'prompt':sequence.native_rl_prompt,'continuations':continuations,
            **({'split_tokenization': True, 'continuation_alternatives': [[list(ids) for ids in alts]
                                                                          for alts in alternatives]} if split else {}),
            'native_state_imports':sequence.native_imports,
            **({'actor_export': dict(sequence.native_actor_export)}
               if getattr(sequence, 'native_actor_export', None) else {}),
            **({'served_policy': dict(sequence.native_served_policy)}
               if getattr(sequence, 'native_served_policy', None) else {})}
    return response


def _score_rendered(engine, item, plan, continuations, bindings, alternatives=None):
    request = decision_request(item, bindings)
    sequence = Sequence(request, None, prompt_plan=plan, native_actor_export=engine.actor_export_default)
    engine._native_prefill(sequence)
    base_cache, base_logits = sequence.cache, sequence.logits
    results = [_score_from_cache(engine, base_cache, base_logits, own)
               for own in decision_sequence_list(continuations, alternatives)]
    return decision_response(item, sequence, continuations, results, alternatives)


def decide_native(engine, body: dict) -> dict:
    """Score candidate replies from a single exact native prompt cache."""
    return _score_item(engine, {"messages": body.get("messages") or [], "tools": body.get("tools"),
                                "options": body.get("options") or [], "native_state_modes": body.get("native_state_modes"),
                                "native_recording":body.get("native_recording",False),
                                **({'open_reply': body['open_reply']} if 'open_reply' in body else {}),
                                **({'reasoning': body['reasoning']} if 'reasoning' in body else {}),
                                **({'split_tokenization': body['split_tokenization']}
                                   if 'split_tokenization' in body else {}),
                                **({'reasoning_ids': body['reasoning_ids']} if 'reasoning_ids' in body else {}),
                                **({'x_natlang_privileged_context': body['x_natlang_privileged_context']}
                                   if body.get('x_natlang_privileged_context') is not None else {})}, body.get('adapters'))


def decide_many_native(engine, body: dict) -> dict:
    """Batch decisions without changing per-item semantics; failures remain item-local."""
    items = body.get("items")
    if not isinstance(items, list):
        raise RequestError("neuralese-decision", "items must be a list")
    def one(item):
        try:
            return _score_item(engine, {"native_state_modes": body.get("native_state_modes"), "native_recording":body.get("native_recording",False), **item}, body.get("adapters"))
        except Exception as error:  # item failures are part of the shared score contract
            # A RequestError's text already starts with its code.
            message = str(error) if isinstance(error, RequestError) else f"internal: {error}"
            return {"error": message[:500]}
    # A batching engine (the SGLang actor) scores the items concurrently, as the llama.cpp fork does; the native engine
    # serializes on its execution lock either way.
    if len(items) > 1 and getattr(engine, 'sglang_score', None) is not None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(items), 16), thread_name_prefix='decide-many') as pool:
            return {"results": list(pool.map(one, items))}
    return {"results": [one(item) for item in items]}
