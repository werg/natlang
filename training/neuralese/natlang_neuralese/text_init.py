"""In-context text initialisation of soft bodies (owner 2026-10-10).

A soft body (a combinator's or soft function's instructions, a system-prompt bank entry, a soft-skill body) sits where
a text-instructed call has its instructions. Its text initialisation is the value under which the soft call **is** the
text-instructed call: the read transport of the body reproduces the input embeddings of the instruction tokens exactly
as the runtime renders and tokenizes them in that call. It is not an encoding of the bare sentence (`/encode`, the
port's value write, is for values a call reads back; spliced into an instruction slot it is a context-free paraphrase
the model never saw there), and not raw token embeddings sent through a read transport that changes them.

`instruction_body` takes the soft call as the runtime renders it, with any placeholder block at the body's position,
and the instruction text. The text-instructed call is the same rendering with the instruction text in place of the
placeholder (a text natlang function's body sits in exactly the field a soft function's body sentinel does). The body
rows are the read transport's inverse of the embeddings of the tokens that the text call's own tokenization gives the
instruction span. `init_gate` then checks that at initialisation the soft call reproduces the text-instructed call:
next-token distributions over the generation prompt and a greedy reply of the text call, teacher-forced through both.

Read transports without markers whose input map is the identity (raw-token-v1, latent-sketch-v1/v2, a read adapter at
its zero initialisation) reproduce the text call exactly when the span tokenizes as it does in the joint rendering.
The legacy RMS profile (read markers and an interface norm) cannot: its gate reports how close it gets.
"""

from __future__ import annotations

import torch

from .serve.chat import escape_specials, render_messages
from .serve.store import make_block

INIT_SCHEMA = "natlang.text-init-in-context/1"


def _read_is_identity(heads, rows: torch.Tensor) -> bool:
    if heads.read_markers:
        return False
    with torch.no_grad():
        back = heads.read_in(rows)
    return bool(torch.equal(back.to(rows.dtype), rows))


def read_inverse(heads, rows: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Payload rows whose read transport gives `rows` (input embeddings, [L, d]), and whether that is exact. The legacy
    interface norm (RMS norm with a gain) is inverted up to its per-row scale: payload `rows / gain`, which the norm
    maps to `rows / rms(rows / gain)`; its read markers stay, so it is never exact."""
    rows = rows.float()
    if not heads.read_markers:
        if _read_is_identity(heads, rows):
            return rows, True
        raise ValueError("the read adapter is no longer the identity: an instruction body has no exact text "
                         "initialisation on these heads (initialise before the adapter trains)")
    gain = getattr(heads.interface, "weight", None)
    if gain is None:
        return rows, False
    return rows / gain.float().to(rows.device), False


def _soft_render(engine, messages, tools, placeholder: str):
    prompt = render_messages(messages, tools, engine._template, engine.specials, block_type=engine.block_value_type)
    where = [i for i, segment in enumerate(prompt.segments) if isinstance(segment, int) and prompt.blocks[segment] == placeholder]
    if len(where) != 1:
        raise ValueError(f"the soft call must hold the placeholder body {placeholder} exactly once, found {len(where)}")
    return prompt, where[0]


def _tokens(engine, segments, nonce) -> list[int]:
    return [i for segment in segments for i in engine._template_tokens(segment, nonce)]


def instruction_span(engine, messages, tools, placeholder: str, text: str) -> dict:
    """The instruction tokens as the text-instructed call tokenizes them. Returns {ids, aligned}: `aligned` when the
    text call's tokens are exactly the soft call's prefix tokens, the span, and its suffix tokens (no merge across the
    span's boundaries), so a body holding the span's embeddings makes the two calls identical."""
    prompt, at = _soft_render(engine, messages, tools, placeholder)
    if any(isinstance(s, int) for s in prompt.segments[:at] + prompt.segments[at + 1:]):
        # Other blocks (arguments, captures) stay blocks in both calls; only the text runs around the body matter.
        pass
    left = [s for s in prompt.segments[:at]]
    right = [s for s in prompt.segments[at + 1:]]
    # Text runs adjacent to the body are tokenized jointly with the instruction text in the text call.
    head = left[-1] if left and isinstance(left[-1], str) else ""
    tail = right[0] if right and isinstance(right[0], str) else ""
    nonce = prompt.escape_nonce
    text = escape_specials(text, engine.specials, nonce)
    joint = engine._template_tokens(head + text + tail, nonce)
    head_ids, tail_ids = engine._template_tokens(head, nonce), engine._template_tokens(tail, nonce)
    aligned = joint[:len(head_ids)] == head_ids and (not tail_ids or joint[len(joint) - len(tail_ids):] == tail_ids)
    if aligned:
        ids = joint[len(head_ids):len(joint) - len(tail_ids)]
    else:
        ids = engine._template_tokens(text, nonce)
    if not ids:
        raise ValueError("empty instruction text")
    return {"ids": ids, "aligned": aligned, "segment_before": len(head), "segment_after": len(tail)}


def instruction_body(engine, messages, tools, placeholder: str, text: str, type: str | None = None):
    """The in-context text initialisation of the body at `placeholder` in the soft call `messages`: registers and
    returns (block, info)."""
    span = instruction_span(engine, messages, tools, placeholder, text)
    with torch.no_grad():
        embeddings = engine.backbone.embed(torch.tensor([span["ids"]], device=engine.device))[0].float()
    rows, exact_transport = read_inverse(engine.heads, embeddings)
    info = {"schema": INIT_SCHEMA, "tokens": len(span["ids"]), "aligned": span["aligned"],
            "exact_transport": exact_transport, "exact": bool(span["aligned"] and exact_transport)}
    block = engine.store.put(make_block(rows, engine.dialect, type=type,
                                        producer={"kind": "text-init-in-context", "text": text, **info}))
    return block, info


def _text_call_items(session, prompt, at: int, text: str):
    segments = list(prompt.segments)
    left = segments[:at]
    right = segments[at + 1:]
    head = left.pop() if left and isinstance(left[-1], str) else ""
    tail = right.pop(0) if right and isinstance(right[0], str) else ""
    text = escape_specials(text, session.engine.specials, prompt.escape_nonce)
    joined = left + [head + text + tail] + right
    return session._items(joined, prompt.blocks, prompt.escape_nonce)


@torch.no_grad()
def init_gate(engine, messages, tools, body_id: str, text: str, *, reply_tokens: int = 24,
              min_agreement: float = 0.98, max_kl: float = 0.02) -> dict:
    """Does the soft call with `body_id` reproduce the text-instructed call at initialisation? Compares next-token
    distributions over the generation prompt's last position and a greedy reply of the text call (`reply_tokens`),
    teacher-forced through both calls."""
    from .serve.grad import GradSession

    session = GradSession(engine)
    prompt, at = _soft_render(engine, messages, tools, body_id)
    soft_items = session._items(prompt.segments, prompt.blocks, prompt.escape_nonce)
    text_items = _text_call_items(session, prompt, at, text)
    backbone = engine.backbone

    def logits(items, reply):
        embeds = session._embed_items(items + [("tok", t) for t in reply], {})
        return backbone.forward_embeds(embeds)["logits"][0].float()

    reply: list[int] = []
    eos = {engine.tokenizer.convert_tokens_to_ids("<|im_end|>"), engine.tokenizer.eos_token_id}
    for _ in range(reply_tokens):
        token = int(logits(text_items, reply)[-1].argmax())
        reply.append(token)
        if token in eos:
            break
    n = len(reply)
    text_logits = logits(text_items, reply)[-(n + 1):-1] if n else logits(text_items, reply)[-1:]
    soft_logits = logits(soft_items, reply)[-(n + 1):-1] if n else logits(soft_items, reply)[-1:]
    t = torch.log_softmax(text_logits, -1)
    s = torch.log_softmax(soft_logits, -1)
    kl = float((t.exp() * (t - s)).sum(-1).mean())
    agreement = float((t.argmax(-1) == s.argmax(-1)).float().mean())
    max_abs = float((text_logits - soft_logits).abs().max())
    return {"schema": "natlang.text-init-gate/1", "positions": int(t.shape[0]), "kl": kl, "agreement": agreement,
            "max_abs_logit_delta": max_abs, "bit_exact": max_abs == 0.0,
            "passed": agreement >= min_agreement and kl <= max_kl,
            "thresholds": {"min_agreement": min_agreement, "max_kl": max_kl},
            "reply_preview": engine.tokenizer.decode(reply)[:200]}


def instruction_rows(engine, text: str, type: str | None = None, messages: list | None = None, tools=None):
    """(block, info) for instruction `text` initialised in context: inside the first message of `messages` whose text
    content holds `text` (a recorded call), else as the whole system message of a minimal call (`info["context"]` says
    which). For Python producers that hold recorded prompts rather than a runtime that renders calls."""
    from .serve.grad import embed_text

    placeholder = embed_text(engine, text, type).id
    context = None
    for index, message in enumerate(messages or []):
        content = message.get("content")
        if isinstance(content, str) and text in content:
            before, after = content.split(text, 1)
            parts = ([{"type": "text", "text": before}] if before else []) + [{"type": "neuralese", "id": placeholder}] + \
                ([{"type": "text", "text": after}] if after else [])
            context = [*messages[:index], {**message, "content": parts}, *messages[index + 1:]]
            break
    where = "recorded-call"
    if context is None:
        where = "system-message"
        context = [{"role": "system", "content": [{"type": "neuralese", "id": placeholder}]},
                   {"role": "user", "content": "Begin."}]
        tools = None
    block, info = instruction_body(engine, context, tools, placeholder, text, type)
    return block, {**info, "context": where}
