"""The builtin `view(value, instructions?)`, its Neuralese instance: write site and plan, shared by the servers and the
trainers (plans/neuralese/DECISIONS.md 2026-10-09: "one summarizer family" and "representation chosen by use").

`view` (ts-host/src/builtin/view.nl) is a representation-generic natlang function: its `string` instance is the crisp
view, its `Neuralese<string>` instance a template write of the same body. Without instructions a view is faithful
compression (trained so that `read(view(x))` reproduces `x`); with instructions it keeps what their purpose needs.

The write site: the body is the system text (prompt piece `view`, or its soft form under a system-prompt bank), the
user message gives the arguments, and the reply is forced by template readout to `return_result(status="success",
value=…)` with the value written as a block (`TEMPLATE`, the cut chat.call_reply makes); the stop head decides the
length. A value that fits the site's window (by default the model's whole context, less the site's own text and the
block) is viewed in one write. A longer value is split into token chunks: each chunk gets a view at a part site
("part i of n"), and a combine site, which reads the part views as blocks, writes the final view. Every write is the
ordinary template write, so in training each one is differentiable and the view's readers train all of them.

The servers (`POST /v1/neuralese/view`) and the trainer (`train.trajectories --view written`) both use this module's
sites, template and `plan`, so what is trained is what runs; tests/fixtures/view-site.json pins the site (the runtime
and the C++ server are checked against it too).

The runtime's opening listing shows the view of each argument it would cut off, written for the receiving call
(`listing_instructions`, ts-host native/prompt.ts `listingViewInstructions`), followed by `view_note`.
"""

from __future__ import annotations

from dataclasses import dataclass

# View's body (prompt piece `view`, ts-host/src/builtin/view.nl; VIEW_PROMPT): the system text of its write site.
INSTRUCTIONS = (
    "Write a view of value: a shorter form of it that a reader uses in its place.\n\n"
    "Without instructions, the view is faithful: keep everything needed to reproduce value exactly, its content, its "
    "structure and its exact details (names, numbers, identifiers, code), and drop only what repeats or can be restated "
    "more briefly without loss.\n\n"
    "With instructions, they say what the view is for: keep what that purpose needs, including how value is organised "
    "and where in it the rest can be found, and leave out what the purpose does not need.")
TYPE = "Neuralese<string>"
# The call the reply is forced to, and its tool: the template write of view's result.
TEMPLATE = {"call": "return_result", "arguments": {"status": "success"}, "argument": "value", "value": "write"}
TOOLS = [{"type": "function", "function": {
    "name": "return_result", "description": "Finish the call: value is the result.",
    "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success"]},
                                                    "value": {"type": "string"}},
                   "required": ["status", "value"]}}}]
SITE_MARGIN = 1024  # tokens kept free beside the value: the site's own text, the tool, the forced reply, the block


def _purpose(instructions: str | None) -> str:
    return f"instructions:\n{instructions}\n\n" if instructions else ""


def view_site(system, value: str, instructions: str | None = None) -> list[dict]:
    """The single write site. `system` is view's body: its text, or parts with its soft form."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{_purpose(instructions)}value:\n{value}"}]


def part_site(system, chunk: str, instructions: str | None, index: int, count: int) -> list[dict]:
    """The write site of one chunk of a value too long for one site."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{_purpose(instructions)}value, part {index + 1} of {count}:\n{chunk}"}]


def combine_site(system, instructions: str | None, parts: list[str]) -> list[dict]:
    """The write site of the final view of a chunked value: the part views (block IDs), in order."""
    content = [{"type": "text", "text": f"{_purpose(instructions)}Views of the {len(parts)} parts of value, in order:\n"}]
    for index, block in enumerate(parts):
        content += [{"type": "text", "text": f"Part {index + 1}: "}, {"type": "neuralese", "id": block},
                    {"type": "text", "text": "\n"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def listing_instructions(name: str, type: str, instructions: str) -> str:
    """The instructions of the view a call's opening listing shows for a value it would cut off."""
    return (f"The view is shown in the opening listing of a call that receives the value as {name} ({type}). The call "
            f"reads {name} itself for exact details, so keep the view short: what the value is, how it is organised, "
            f"and what in it bears on the call's instructions, so that the call knows what to look up. The call's "
            f"instructions:\n{instructions}")


def view_note(holder: str) -> str:
    return f"  // view of the value; {holder} holds all of it"


def reply_prefix(apply_template) -> str:
    """The forced reply before the written value: the model's own rendering of `return_result(status="success",
    value="`, as the template readout cuts it (`apply_template(messages, add_generation_prompt)`)."""
    from .serve.chat import write_reply

    return write_reply(apply_template, TEMPLATE["call"], TEMPLATE["arguments"], TEMPLATE["argument"], "string")[0]


@dataclass
class Plan:
    chunks: list[str]  # one entry: the whole value, viewed at the single site

    @property
    def chunked(self) -> bool:
        return len(self.chunks) > 1


def plan(tokenizer, value: str, window: int) -> Plan:
    """Split `value` into chunks of at most `window` tokens (one chunk when it fits)."""
    ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    if len(ids) <= window:
        return Plan([value])
    return Plan([tokenizer.decode(ids[start:start + window]) for start in range(0, len(ids), window)])


def window_of(engine, instructions: str | None, requested: int | None = None) -> int:
    """Value tokens one write site holds: the model's context less the site's text, tool, reply and block (or less)."""
    context = int(getattr(engine.backbone.hf.config, "max_position_embeddings", 32768))
    overhead = len(engine.tokenizer(instructions or "", add_special_tokens=False)["input_ids"]) + engine.max_block + SITE_MARGIN
    available = max(256, context - overhead)
    return min(available, requested) if requested else available


def write_view(write, system, value: str, instructions: str | None, tokenizer, window: int):
    """Run the plan with `write(messages) -> block ID` (a template write at the site): the final view's block ID and
    the number of parts."""
    steps = plan(tokenizer, value, window)
    if not steps.chunked:
        return write(view_site(system, value, instructions)), 1
    parts = [write(part_site(system, chunk, instructions, i, len(steps.chunks)))
             for i, chunk in enumerate(steps.chunks)]
    return write(combine_site(system, instructions, parts)), len(parts)


def reject_digest_part(part: dict) -> None:
    """Records converted before the rename carry `digest` parts; they are converted, never read as views silently."""
    if part.get("type") == "digest":
        raise ValueError(
            "a record part has type 'digest': records converted before natlang.neuralese-conversion/15 and "
            "natlang.harness-bench-conversion/2 name the listing view 'digest' (DECISIONS.md 2026-10-09, one summarizer "
            "family: view). Convert the records with scripts/neuralese_data/digest_to_view.py, which rewrites each "
            "digest part as a view part and the piece prompt:digest as prompt:view, and register the converted corpus "
            "with its own manifest; the original stays as published.")
