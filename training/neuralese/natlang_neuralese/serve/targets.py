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


def target_items(engine, messages, tools, target):
    """Return (prompt items, target continuation items) under the runtime's exact chat rendering."""
    prompt = engine.render_request(messages, tools)
    thinking = opens_thinking(engine._template([{"role": "user", "content": "x"}], None))

    def full_template(msgs, tls):
        render = lambda values: engine.tokenizer.apply_chat_template(
            values, tools=tls or None, tokenize=False, add_generation_prompt=False)
        return render_with_empty_thought(render, msgs) if thinking else render(msgs)

    full = engine.render_request(list(messages) + [target], tools, template=full_template)
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
