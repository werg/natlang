"""Prompt rendering modes of native training and serving documents (plans/neuralese/NO_SYSTEM_PROMPT_RENDERING_DECOMPOSITION.md).

``full``: the runtime's system message and tool schemas, as recorded.
``program-header``: the GENERIC runtime instructions and the tool schemas are dropped (owner decision 2026-10-11:
distil the prompt into the weights after a warm-up); the system paragraphs that declare a property of THIS call
are kept verbatim as a short system turn, the program header. A call with none of them has no system turn.

The scope table is ts-host's ``promptPieces()`` with its ``scope`` field plus the historical texts of every piece
(recorded corpora carry earlier runtime prompts), exported to ``training/prompt-pieces/pieces.json``; the algorithm
is a port of ts-host ``renderSystemPrompt`` (parity: tests/neuralese/test_prompt_rendering_parity.py).
``PROGRAM_SPECIFIC_HEADS`` was the interim paragraph-head table (kept for reference; no longer used).
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


_PIECES = None


def default_pieces() -> list:
    """The shared scope table: ts-host promptPieces() plus the historical bank (training/prompt-pieces/pieces.json,
    exported by ts-host/scripts/export-prompt-pieces.mjs --history)."""
    global _PIECES
    if _PIECES is None:
        from pathlib import Path
        _PIECES = load_pieces(Path(__file__).resolve().parents[4] / 'training' / 'prompt-pieces' / 'pieces.json')
    return _PIECES


def apply_prompt_rendering(messages, tools, prompt_rendering: str = "full", pieces=None):
    """``(messages, tools)`` for ``prompt_rendering``. ``full`` returns them unchanged; ``program-header`` renders the
    system message with ts-host's renderSystemPrompt algorithm (render_system_prompt) and passes no tool schemas."""
    if prompt_rendering not in PROMPT_RENDERINGS:
        raise ValueError(f"unknown prompt_rendering {prompt_rendering!r} (declared: {PROMPT_RENDERINGS})")
    if prompt_rendering == "full":
        return messages, tools
    out = list(messages)
    if out and isinstance(out[0], dict) and out[0].get("role") == "system":
        content = out[0].get("content")
        if not isinstance(content, str):
            raise ValueError("program-header rendering needs a plain-text system message")
        header = render_system_prompt(content, prompt_rendering, default_pieces() if pieces is None else pieces)
        out = ([{**out[0], "content": header}] if header else []) + out[1:]
    if any(isinstance(m, dict) and m.get("role") == "system" for m in out[1:]):
        raise ValueError("program-header rendering found a system message after the first turn")
    return out, None


# --- Port of ts-host renderSystemPrompt (src/native/system-prompts.ts, 18dda11d) over the exported piece table. ---
# The table is ts-host's promptPieces() (id, scope, text) exported by ts-host/scripts/export-prompt-pieces.mjs to
# training/prompt-pieces/pieces.json, later extended with the historical texts of each generic piece (D9). Parity
# with the TS function is checked on recorded system prompts (tests/neuralese/test_prompt_rendering_parity.py).

def find_pieces(text: str, pieces) -> list:
    """findPieces: occurrences of each piece's text, longest pieces first, without overlaps, sorted by start."""
    found = []

    def taken(start, end):
        return any(start < e and s < end for s, e, _ in found)
    for piece in sorted(pieces, key=lambda p: -len(p['text'])):  # stable, as Array.prototype.sort
        needle = piece['text']
        if not needle:
            continue
        at = text.find(needle)
        while at >= 0:
            if not taken(at, at + len(needle)):
                found.append((at, at + len(needle), piece))
            at = text.find(needle, at + len(needle))
    return sorted(found, key=lambda item: item[0])


def render_system_prompt(text: str, mode: str = 'full', pieces=()) -> str:
    """renderSystemPrompt: drop generic pieces, keep program pieces nested in them, keep unmatched text verbatim."""
    if mode not in PROMPT_RENDERINGS:
        raise ValueError(f"unknown prompt_rendering {mode!r} (declared: {PROMPT_RENDERINGS})")
    if mode == 'full':
        return text
    generic = find_pieces(text, [p for p in pieces if p['scope'] == 'generic'])
    program = find_pieces(text, [p for p in pieces if p['scope'] == 'program'])
    out, cursor = '', 0
    for start, end, _ in generic:
        out += text[cursor:start]
        retained = [text[a:b] for a, b, _ in program if a >= start and b <= end]
        if retained:
            out += '\n\n' + '\n\n'.join(retained) + '\n\n'
        cursor = end
    return (out + text[cursor:]).strip(' \t\n\r\x0b\x0c ﻿  ')


def load_pieces(path) -> list:
    import json
    from pathlib import Path
    table = json.loads(Path(path).read_text(encoding='utf-8'))
    return [p for p in table['pieces'] if p.get('scope') in ('generic', 'program') and isinstance(p.get('text'), str)]
