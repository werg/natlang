"""E2E cohort rendering of converted native records (plans/neuralese/E2E_DATA.md).

Kept out of ``data/prompt_rendering.py`` so helpers built on the shared renderer never change the prompt contract
module itself (its behavioural identity is ``prompt_rendering.prompt_rendering_identity``).
"""
from __future__ import annotations

from .prompt_rendering import apply_prompt_rendering


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

