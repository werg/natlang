"""Prompt rendering modes of native training and serving documents (plans/neuralese/NO_SYSTEM_PROMPT_RENDERING_DECOMPOSITION.md).

``full``: the runtime's system message and tool schemas, as recorded.
``program-header``: the GENERIC runtime instructions and the tool schemas are dropped (owner decision 2026-10-11:
distil the prompt into the weights after a warm-up); the system paragraphs that declare a property of THIS call
are kept verbatim as a short system turn, the program header. A call with none of them has no system turn.

``PROGRAM_SPECIFIC_HEADS`` is the interim scope table until ts-host exports ``promptPieces()`` with a ``scope``
field (decomposition step 1); ``scripts/neuralese_data/strip_generic_system_prompt.py`` uses the same table on
rendered text, and the two are checked against each other byte for byte.
"""
from __future__ import annotations

PROMPT_RENDERINGS = ("full", "program-header")

# Paragraph heads (exact prefixes) that declare a property of THIS call rather than generic runtime behaviour.
PROGRAM_SPECIFIC_HEADS = (
    'This call is a directory reducer:',
    'This call is at the ',  # nesting layer of ad hoc natural-language calls
    '\nDeclared Neuralese text-channel emulation for this run:',
    'Declared Neuralese text-channel emulation for this run:',
    'At a direct parameter declared Neuralese<string>',
    "When this call's declared result is exactly Neuralese<string>",
    'If a function you call is unclear or wrong, read it with read_code and fix it with edit_code',
)


def program_header_paragraphs(prose: str) -> list[str]:
    """The program-specific paragraphs of a system prompt's prose, in order and verbatim."""
    return [p for p in prose.split('\n\n') if p.startswith(PROGRAM_SPECIFIC_HEADS)]


def apply_prompt_rendering(messages, tools, prompt_rendering: str = "full"):
    """``(messages, tools)`` for ``prompt_rendering``. ``full`` returns them unchanged."""
    if prompt_rendering not in PROMPT_RENDERINGS:
        raise ValueError(f"unknown prompt_rendering {prompt_rendering!r} (declared: {PROMPT_RENDERINGS})")
    if prompt_rendering == "full":
        return messages, tools
    out = list(messages)
    if out and isinstance(out[0], dict) and out[0].get("role") == "system":
        content = out[0].get("content")
        if not isinstance(content, str):
            raise ValueError("program-header rendering needs a plain-text system message")
        kept = program_header_paragraphs(content)
        out = ([{**out[0], "content": "\n\n".join(kept)}] if kept else []) + out[1:]
    if any(isinstance(m, dict) and m.get("role") == "system" for m in out[1:]):
        raise ValueError("program-header rendering found a system message after the first turn")
    return out, None
