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


def prompt_rendering_identity(mode: str) -> dict:
    """Content identity for the serving/training prompt transformation contract."""
    if mode not in PROMPT_RENDERINGS:
        raise ValueError(f"unknown prompt_rendering {mode!r} (declared: {PROMPT_RENDERINGS})")
    import hashlib
    import json
    from pathlib import Path

    source = Path(__file__).resolve()
    package_root = source.parents[1]
    pieces_path = source.parents[4] / "training" / "prompt-pieces" / "pieces.json"
    implementation = {name: hashlib.sha256((package_root / name).read_bytes()).hexdigest()
                       for name in ("data/prompt_rendering.py", "serve/engine.py", "serve/chat.py")}
    pieces_sha = hashlib.sha256(pieces_path.read_bytes()).hexdigest()
    identity = {"schema": "natlang.prompt-rendering-identity/1", "mode": mode,
                "implementation_sha256": implementation, "pieces_sha256": pieces_sha}
    identity["identity_sha256"] = hashlib.sha256(
        (identity["schema"] + "\0" + mode + "\0" + json.dumps(implementation, sort_keys=True) +
         "\0" + pieces_sha).encode()).hexdigest()
    return identity


def checkpoint_prompt_rendering(identity: dict, *, serving_mode: str | None = None) -> tuple[str, dict]:
    """Resolve serving mode from the exact direct-SFT text input pinned by checkpoint identity.

    New rendered rows declare their mode. Older inputs without a declaration historically use the full prompt.
    Mixed sources require an explicit serving policy; the returned proof keeps declared and undeclared rows
    separate so the policy choice does not rewrite the training source's provenance.
    """
    import hashlib
    import json
    from pathlib import Path

    options = identity.get("options") or {}
    inputs = identity.get("inputs") or {}
    if serving_mode is not None and serving_mode not in PROMPT_RENDERINGS:
        raise ValueError(f"unsupported explicit serving prompt_rendering {serving_mode!r}")
    text_value = options.get("text_data")
    if not isinstance(text_value, str) or not text_value:
        # A direct stage sourced from records/pieces rather than prepared text uses the legacy full prompt.
        mode = serving_mode or options.get("prompt_rendering", "full")
        if mode not in PROMPT_RENDERINGS:
            raise ValueError(f"checkpoint declares unsupported prompt_rendering {mode!r}")
        return mode, {"schema": "natlang.prompt-rendering-source-proof/1", "source": "legacy-no-text-data",
                      "mode": mode, "mode_selection": "explicit-serving-policy" if serving_mode else "legacy-default",
                      "row_count": None, "mode_counts": {mode: None}, "declared_mode_counts": {},
                      "undeclared_mode_rows": None,
                      "input_sha256": None}
    path = Path(text_value).resolve(strict=True)
    declared_sha = inputs.get(str(path))
    if not isinstance(declared_sha, str):
        # Some checkpoints record the same path before a symlink was resolved.
        declared_sha = next((value for key, value in inputs.items()
                             if Path(key).resolve() == path), None)
    if not isinstance(declared_sha, str):
        raise ValueError("checkpoint identity does not bind its text_data input")
    digest = hashlib.sha256()
    counts = {}
    declared_counts = {}
    system_mode_counts = {}
    rows_with_tool_schemas = 0
    undeclared_rows_with_tool_schemas = 0
    row_count = 0
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            row_count += 1
            mode = row.get("prompt_rendering")
            if mode is not None:
                if mode not in PROMPT_RENDERINGS:
                    raise ValueError(f"text_data row {row_count} declares unsupported prompt_rendering {mode!r}")
                declared_counts[mode] = declared_counts.get(mode, 0) + 1
            else:
                mode = "full"
            if mode not in PROMPT_RENDERINGS:
                raise ValueError(f"text_data row {row_count} declares unsupported prompt_rendering {mode!r}")
            counts[mode] = counts.get(mode, 0) + 1
            system_mode = row.get("system_prompt_mode")
            system_mode_key = system_mode if isinstance(system_mode, str) else "absent"
            system_mode_counts[system_mode_key] = system_mode_counts.get(system_mode_key, 0) + 1
            if "List of tools:" in str(row.get("text", "")):
                rows_with_tool_schemas += 1
                if row.get("prompt_rendering") is None:
                    undeclared_rows_with_tool_schemas += 1
    actual_sha = digest.hexdigest()
    if actual_sha != declared_sha:
        raise ValueError("checkpoint text_data changed since training")
    mixed = len(counts) != 1
    if mixed and serving_mode is None:
        raise ValueError(f"checkpoint text_data mixes prompt rendering modes and has no explicit serving policy: {counts}")
    source_mode = next(iter(counts), "full") if not mixed else None
    mode = serving_mode or source_mode or "full"
    if not mixed and serving_mode is not None and serving_mode != source_mode:
        raise ValueError("explicit serving prompt_rendering differs from homogeneous text_data rows")
    option_mode = options.get("prompt_rendering")
    if option_mode is not None and option_mode != mode:
        raise ValueError("checkpoint prompt_rendering option differs from the selected serving mode")
    proof = {"schema": "natlang.prompt-rendering-source-proof/1", "source": "checkpoint-identity-text-data",
             "path": str(path), "input_sha256": declared_sha, "row_count": row_count,
             "mode_counts": counts, "declared_mode_counts": declared_counts,
             "undeclared_mode_rows": row_count - sum(declared_counts.values()),
             "system_prompt_mode_counts": system_mode_counts,
             "rows_with_tool_schema_marker": rows_with_tool_schemas,
             "undeclared_rows_with_tool_schema_marker": undeclared_rows_with_tool_schemas,
             "mode": mode,
             "mode_selection": ("explicit-serving-policy-on-mixed-training-input" if mixed else
                                "explicit-serving-policy" if serving_mode else "homogeneous-source")}
    return mode, proof

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


def program_header_native_record(record: dict, record_pieces, pieces=None) -> dict:
    """A converted native record (``natlang.teacher_training_turn.native/1``) in ``program-header`` rendering.

    The leading system message's parts (soft prompt pieces resolved through ``record_pieces``, name -> text, plus text
    parts) become its prose, which ``apply_prompt_rendering`` reduces to the program header; the system turn is dropped
    when no header remains, and the runtime tool schemas are dropped (the renderer passes none). Everything else, i.e.
    the user/assistant/tool messages with their soft, read and neuralese parts, the target and every provenance field,
    is unchanged, so producer->consumer edges authenticate exactly as in the full record. The record declares
    ``prompt_rendering`` and ``system_prompt_mode``.
    """
    messages = list(record.get("messages") or [])
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "soft":
                    texts.append(record_pieces[part["name"]])
                elif kind == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                else:
                    raise ValueError(f"system message part {kind!r} has no program-header rendering")
            messages[0] = {**messages[0], "content": "".join(texts)}
    rendered, _ = apply_prompt_rendering(messages, record.get("tools"), "program-header", pieces)
    has_header = bool(rendered) and rendered[0].get("role") == "system"
    return {**record, "messages": rendered, "tools": [], "prompt_rendering": "program-header",
            "system_prompt_mode": "program-own" if has_header else "none"}


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
