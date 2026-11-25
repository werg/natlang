"""Shared chat-template tokenization for teacher-forced target continuations."""

from .chat import RequestError, opens_thinking, render_with_empty_thought


def _items(engine, segments, blocks, escape_nonce="", block_stop_gradients=None, block_source_occurrences=None):
    items = []
    for segment in segments:
        if isinstance(segment, str):
            items.extend(("tok", token) for token in engine._template_tokens(segment, escape_nonce))
        else:
            occurrence = block_source_occurrences[segment] if block_source_occurrences else None
            value = {"id": blocks[segment], "source_occurrence": occurrence} if occurrence else blocks[segment]
            is_constant = bool(block_stop_gradients and block_stop_gradients[segment])
            items.append(("block_stop" if is_constant else "block", value))
    return items


def _split_boundary(engine, prefix, after):
    """Recover target text when tokenization merges the prompt's last token with its first token."""
    head, last = prefix[:-1], prefix[-1]
    if last[0] != "tok":
        return None
    decode = lambda token: engine.tokenizer.decode([token])
    tail = decode(last[1])
    for index in range(len(after) - len(prefix), -1, -1):
        joined = after[index + len(head)] if index + len(head) < len(after) else None
        if after[index:index + len(head)] == head and joined is not None and joined[0] == "tok":
            text = decode(joined[1])
            if text.startswith(tail) and len(text) > len(tail):
                return [("tok", token) for token in engine._tokens(text[len(tail):])] + after[index + len(head) + 1:]
    return None


def _closed_template(engine):
    """The chat template without a generation prompt: complete messages, as a later prompt renders them."""
    thinking = opens_thinking(engine._template([{"role": "user", "content": "x"}], None))

    def full_template(msgs, tls):
        render = lambda values: engine.tokenizer.apply_chat_template(
            values, tools=tls or None, tokenize=False, add_generation_prompt=False)
        return render_with_empty_thought(render, msgs) if thinking else render(msgs)
    return full_template


def opening_items(engine, messages, tools):
    """The items of `messages` rendered as the opening of a longer prompt (no generation prompt): the token prefix a
    prompt that continues these messages starts with, when the template renders append-only (callers check)."""
    rendered = engine.render_request(list(messages), tools, template=_closed_template(engine))
    return _items(engine, rendered.segments, rendered.blocks, rendered.escape_nonce,
                  rendered.block_stop_gradients, rendered.block_source_occurrences)


def target_items(engine, messages, tools, target, *, response_tools=None):
    """Return (prompt items, target continuation items) under the runtime's exact chat rendering."""
    prompt = engine.render_request(messages, tools, **({'response_tools':response_tools} if response_tools is not None else {}))
    full = engine.render_request(list(messages) + [target], tools, template=_closed_template(engine),
                                 **({'response_tools':response_tools} if response_tools is not None else {}))
    before = _items(engine, prompt.segments, prompt.blocks, prompt.escape_nonce,
                    prompt.block_stop_gradients, prompt.block_source_occurrences)
    after = _items(engine, full.segments, full.blocks, full.escape_nonce,
                   full.block_stop_gradients, full.block_source_occurrences)
    im_start = engine.tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = [index for index, item in enumerate(before) if item == ("tok", im_start)]
    if not starts:
        raise RequestError("neuralese-grad-target", "the rendered prompt has no generation prefix")
    prefix = before[starts[-1]:]
    found = [index for index in range(len(after) - len(prefix), -1, -1)
             if after[index:index + len(prefix)] == prefix]
    if found:
        rest = after[found[0] + len(prefix):]
    else:
        rest = _split_boundary(engine, prefix, after)
        if rest is None:
            raise RequestError("neuralese-grad-target", "the target's rendering has no assistant prefix")
    im_end = engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = [index for index, item in enumerate(rest) if item == ("tok", im_end)]
    if ends:
        rest = rest[:ends[-1] + 1]
    return before, rest


def non_student_context(tokenizer, target_message, target_ids, *, think="context", lead_in="context"):
    """Leading tokens of a rendered reply the student did not sample (teacher, October, live, a privileged run) that
    are context, not loss: its think block through ``</think>`` and the whitespace after it, and (``lead_in``
    'context') the text it writes before its first tool call. ``think`` 'train' keeps visible reasoning (and the text
    after it) as target; an empty think block stays context. A reply without tool calls (an answer) trains after its
    think block. Teacher think blocks are summaries at a teacher's length (held reasoning p50 fell 1,919 -> 350
    characters after sft-v3 trained them) and lead-in text is narration (Claude 28% of tool calls, held after sft-v3
    12% vs the base's 2%); no verified outcome of the student selects either (DECISIONS 2026-11-18)."""
    if think not in ("context", "train") or lead_in not in ("context", "train"):
        raise ValueError(f"think {think!r} / lead_in {lead_in!r}: context or train")
    if think == "train" and (target_message.get("reasoning_content") or "").strip():
        return 0
    ids = list(target_ids)
    end = think_end(tokenizer, ids)
    if lead_in == "context" and target_message.get("tool_calls") and (target_message.get("content") or "").strip():
        start = _first(tokenizer, ids, "<tool_call>", "<|tool_call_start|>")
        if start is not None:
            end = max(end, start)
    return end


def _first(tokenizer, ids, *tokens):
    """The first position in ``ids`` of any of these vocabulary tokens (templates name them differently), or None."""
    found = [ids.index(i) for i in (tokenizer.convert_tokens_to_ids(t) for t in tokens)
             if isinstance(i, int) and i in ids and tokenizer.decode([i]) in tokens]
    return min(found) if found else None


def think_end(tokenizer, target_ids):
    """The end of a rendered reply's think block: the position after ``</think>`` and the whitespace after it (0 when
    the reply has none). ``non_student_context`` masks up to here; a weighted teacher think block
    (``sft_sources.think_context`` ``train@W``) carries weight W up to here."""
    ids = list(target_ids)
    close = _first(tokenizer, ids, "</think>")
    if close is None:
        return 0
    end = close + 1
    while end < len(ids) and not tokenizer.decode([ids[end]]).strip():
        end += 1
    return end
