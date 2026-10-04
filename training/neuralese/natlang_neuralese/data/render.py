"""Per-model rendering of port records and text spans into token sequences (S1 §2.3, S3 §5).

Structure (role markers, block markers) is inserted as special-token IDs; content is
tokenized with special-token splitting disabled, so marker text inside ordinary content
stays text (S0 §3.3). Rendering never edits records.

A rendered port example has three views of one record:

- **producer**: the write site, ending with the open marker. The writer's instructions,
  declared result type and causal context precede the marker; the purpose lives there.
- **consumer**: token segments before and after the block, then the target. Sources the
  record withholds are absent, so the consumer must use the channel.
- **teacher**: the consumer view with the full source text in place of the block, for
  self-distillation (the crisp base, deltas off).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .records import PortRecord

FORMS = ("chat", "natlang")


@dataclass(frozen=True)
class SpecialIds:
    bos: int
    im_start: int
    im_end: int
    tool_call_start: int
    tool_call_end: int
    open: int
    close: int

    @staticmethod
    def from_tokenizer(tokenizer, controls) -> "SpecialIds":
        ids = tokenizer.convert_tokens_to_ids(
            ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|tool_call_start|>", "<|tool_call_end|>"])
        return SpecialIds(*ids, open=controls.open_id, close=controls.close_id)


class Renderer:
    def __init__(self, tokenizer, controls):
        self.tokenizer = tokenizer
        self.special = SpecialIds.from_tokenizer(tokenizer, controls)

    def text(self, value: str) -> list[int]:
        """Content tokens: marker text in content is never a control token."""
        if not value:
            return []
        return self.tokenizer(value, add_special_tokens=False, split_special_tokens=True).input_ids

    def turn_open(self, role: str) -> list[int]:
        return [self.special.im_start, *self.text(f"{role}\n")]

    def turn_close(self) -> list[int]:
        return [self.special.im_end, *self.text("\n")]

    def chat(self, messages: list[dict], generation_prompt: bool = False) -> list[int]:
        """The LFM2 chat template, built from IDs (tested against `apply_chat_template`)."""
        ids = [self.special.bos]
        for message in messages:
            ids += self.turn_open(message["role"]) + self.text(message["content"]) + self.turn_close()
        if generation_prompt:
            ids += self.turn_open("assistant")
        return ids


@dataclass
class RenderedRecord:
    record_id: str
    family: str
    producer: list[int]           # ends with the open marker
    consumer_before: list[int]    # ends just before the open marker
    consumer_after: list[int]     # begins just after the close marker
    target: list[int]             # loss positions, appended after consumer_after
    teacher_prefix: list[int]     # consumer view with the full source instead of the block
    source_tokens: int


def variable_name(result_type: str) -> str:
    inner = re.sub(r"^Neuralese<|>$", "", result_type).split(",")[0].strip()
    word = re.sub(r"[^A-Za-z0-9]", "", inner) or "value"
    return word[0].lower() + word[1:]


def render_record(renderer: Renderer, record: PortRecord, form: str = "chat") -> RenderedRecord:
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    sp = renderer.special
    text = renderer.text
    name = variable_name(record.result_type)
    source = record.source_text()
    sources_withheld = "sources" in record.withheld

    # Producer: the write site. Purpose comes from instructions, type and context.
    writer_messages = [*record.writer_context,
                       {"role": "user", "content": f"{record.writer_instructions}\n\n{source}"}]
    producer = renderer.chat(list(writer_messages), generation_prompt=True)
    declaration = f"const {name}: {record.result_type} = "
    if form == "chat":
        producer += text(declaration) + [sp.open]
    else:
        producer += [sp.tool_call_start] + text(f'[eval(code="{declaration}') + [sp.open]

    # Consumer: the block appears as a typed declaration in the opening of its view.
    context = [dict(m) for m in record.consumer_context] or [{"role": "user", "content": ""}]
    first_user = next((i for i, m in enumerate(context) if m["role"] == "user"), 0)
    prior, rest = context[:first_user], context[first_user:]
    user_content = rest[0]["content"]
    visible_source = "" if sources_withheld else f"{source}\n\n"
    before = renderer.chat(prior) + renderer.turn_open(rest[0]["role"])
    if form == "chat":
        before += text(f"{visible_source}{declaration}") + [sp.open]
    else:
        # Natlang form: the value is listed in the call's opening scope as an eagerly typed declaration.
        before += text(f"{visible_source}```ts\n{declaration}") + [sp.open]
    after = [sp.close] + text(";\n```\n\n" if form == "natlang" else ";\n\n")
    after += text(user_content) + renderer.turn_close()
    for message in rest[1:]:
        after += renderer.turn_open(message["role"]) + text(message["content"]) + renderer.turn_close()
    after += renderer.turn_open("assistant")
    target = text(record.target) + [sp.im_end]

    # Teacher: same view with the full source text where the block was.
    teacher = renderer.chat(prior) + renderer.turn_open(rest[0]["role"])
    teacher += text(f"{source}\n\n{user_content}") + renderer.turn_close()
    for message in rest[1:]:
        teacher += renderer.turn_open(message["role"]) + text(message["content"]) + renderer.turn_close()
    teacher += renderer.turn_open("assistant")

    return RenderedRecord(record.id, record.family, producer, before, after, target, teacher,
                          source_tokens=len(text(source)))


@dataclass
class SpanExample:
    """Ordinary text with one designated span (phases A–C).

    The plain sequence is prefix + span + continuation. The block form is
    prefix + <open> + block(span) + <close> + continuation.
    """

    prefix: list[int]
    span: list[int]
    continuation: list[int]


def span_examples(renderer: Renderer, texts, prefix_len: int, span_len: int, cont_len: int,
                  stride: int | None = None, limit: int | None = None, span_lengths=None):
    """Windows of a fixed prefix and continuation around a span.

    `span_lengths` (a sequence) varies the span length window by window, cycling through it, so the stop head
    cannot learn a single count; the trainer batches equal lengths together, so batches need no padding. Without
    it every span has `span_len` tokens.

    Every prefix starts with the BOS token: LFM2 relies on it as an attention sink, and a
    window cut from the middle of a document without it is badly mispredicted (8-11 nats
    per token instead of 2-4 on ordinary prose).
    """
    lengths = list(span_lengths) if span_lengths else [span_len]
    count = 0
    bos = renderer.special.bos
    for value in texts:
        ids = renderer.text(value)
        start = 0
        while True:
            span = lengths[count % len(lengths)]
            window = prefix_len + span + cont_len
            if start + window > len(ids):
                break
            chunk = ids[start: start + window]
            yield SpanExample([bos] + chunk[:prefix_len], chunk[prefix_len: prefix_len + span],
                              chunk[prefix_len + span:])
            start += stride or window
            count += 1
            if limit is not None and count >= limit:
                return
