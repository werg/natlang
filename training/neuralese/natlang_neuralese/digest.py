"""The digest operator (DECISIONS.md 43): its write sites and its plan, shared by the server and the trainers.

The runtime lists a large argument as a digest the model writes from the full value at a write site conditioned on the
receiving call's instructions. A value that fits the write site's window (by default the model's whole context, less
the site's own text and the block) is digested in one write. A longer value is split into token chunks: each chunk
gets a digest at a part site ("part i of n"), and a combine site, which reads the part digests as blocks, writes the
final digest. Every write is the ordinary write procedure, so in training each one is differentiable and the
listing's consumers train all of them.

The server (`POST /v1/neuralese/digest`) and the trainer (`train.trajectories --digest written`) both use `plan`, so
what is trained is what runs; tests/fixtures/digest-site.json pins the single-site text (the C++ server checks against
it too).
"""

from __future__ import annotations

from dataclasses import dataclass

# The digest instructions (prompt piece `digest`, ts-host/src/native/prompt.ts DIGEST_PROMPT); the soft form replaces them
# under a system-prompt bank.
INSTRUCTIONS = "Write a digest of the value below for the listing of the call that receives it: what the value is, how it is organised, and what in it bears on the call's instructions, so that the call knows what to look up. The full value stays in the call's variable for exact reading, so do not copy exact details; keep the digest short."
PREFIX = "const digest: Neuralese<Digest> = "
TYPE = "Neuralese<Digest>"
SITE_MARGIN = 512  # tokens kept free beside the value: the site's own text, the prefix, the block


def digest_site(system, name: str, type: str, value: str, instructions: str) -> list[dict]:
    """The single write site. `system` is the digest instructions: their text, or parts with their soft form."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"The call that receives the value has these instructions:\n{instructions}\n\n"
                                        f"The value of {name} ({type}):\n{value}"}]


def part_site(system, name: str, type: str, chunk: str, instructions: str, index: int, count: int) -> list[dict]:
    """The write site of one chunk of a value too long for one site."""
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"The call that receives the value has these instructions:\n{instructions}\n\n"
                                        f"Part {index + 1} of {count} of the value of {name} ({type}):\n{chunk}"}]


def combine_site(system, name: str, type: str, instructions: str, parts: list[str]) -> list[dict]:
    """The write site of the final digest of a chunked value: the part digests (block IDs), in order."""
    content = [{"type": "text", "text": f"The call that receives the value has these instructions:\n{instructions}\n\n"
                                        f"Digests of the {len(parts)} parts of the value of {name} ({type}), in order:\n"}]
    for index, block in enumerate(parts):
        content += [{"type": "text", "text": f"Part {index + 1}: "}, {"type": "neuralese", "id": block},
                     {"type": "text", "text": "\n"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def digest_note(holder: str) -> str:
    return f"  // digest of the value; {holder} holds all of it"


@dataclass
class Plan:
    chunks: list[str]  # one entry: the whole value, digested at the single site

    @property
    def chunked(self) -> bool:
        return len(self.chunks) > 1


def plan(tokenizer, value: str, window: int) -> Plan:
    """Split `value` into chunks of at most `window` tokens (one chunk when it fits)."""
    ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    if len(ids) <= window:
        return Plan([value])
    return Plan([tokenizer.decode(ids[start:start + window]) for start in range(0, len(ids), window)])


def window_of(engine, instructions: str, requested: int | None = None) -> int:
    """Value tokens one write site holds: the model's context less the site's text, prefix and block (or less)."""
    context = int(getattr(engine.backbone.hf.config, "max_position_embeddings", 32768))
    overhead = len(engine.tokenizer(instructions, add_special_tokens=False)["input_ids"]) + engine.max_block + SITE_MARGIN
    available = max(256, context - overhead)
    return min(available, requested) if requested else available


def write_digest(write, system, name: str, type: str, value: str, instructions: str, tokenizer, window: int):
    """Run the plan with `write(messages) -> block ID`: the final digest's block ID and the number of parts."""
    steps = plan(tokenizer, value, window)
    if not steps.chunked:
        return write(digest_site(system, name, type, value, instructions)), 1
    parts = [write(part_site(system, name, type, chunk, instructions, i, len(steps.chunks)))
             for i, chunk in enumerate(steps.chunks)]
    return write(combine_site(system, name, type, instructions, parts)), len(parts)
