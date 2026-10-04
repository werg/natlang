"""The soft system-prompt bank in training (plans/neuralese/DECISIONS.md 40).

A bank is a `.nz` file with one export, `systemPrompts`: a record from prompt-piece ID to a `Neuralese<SystemPrompt>`
block (ts-host/scripts/neuralese-system-prompt-bank.mjs builds the text-initialised one). Each block's producer keeps
the text the piece was initialised from (`text` for a text-initialised block, `init_text` once trained), which is
also the text the piece replaces in a conversation.

`soften` mirrors the runtime's softening (ts-host/src/native/system-prompts.ts): in every non-assistant message with
string content, each piece's exact text, longest pieces first and without overlaps, becomes a block part. Training
passes the pieces' parameters as gradient leaves under their block IDs, so they train like any other leaf; `save_bank`
writes the trained pieces as a new bank in the same format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .serve.store import make_block

EXPORT = "systemPrompts"
TYPE = "Neuralese<SystemPrompt>"


@dataclass
class PromptBank:
    dialect: str
    texts: dict[str, str]  # piece ID → the text it replaces
    rows: dict[str, torch.Tensor]  # piece ID → [length, width] float32
    ids: dict[str, str] = field(default_factory=dict)  # piece ID → block ID in the loaded file
    provenance: dict = field(default_factory=dict)


def load_bank(path) -> PromptBank:
    with safe_open(str(path), framework="pt") as stream:
        header = json.loads(stream.metadata()["natlang"])
        values = header["exports"][EXPORT]["value"]
        texts, rows, ids = {}, {}, {}
        for piece, ref in values.items():
            block_id = ref["$neuralese"]["id"]
            producer = header["blocks"][block_id].get("producer") or {}
            text = producer.get("init_text", producer.get("text"))
            if not isinstance(text, str):
                raise ValueError(f"{path}: piece {piece} does not record the text it replaces")
            texts[piece], rows[piece], ids[piece] = text, stream.get_tensor(block_id).float(), block_id
    return PromptBank(header["dialect"], texts, rows, ids, header.get("provenance") or {})


def find_pieces(text: str, texts: dict[str, str]) -> list[tuple[int, int, str]]:
    """(start, end, piece ID) of piece occurrences, longest pieces first, without overlaps, in text order."""
    found: list[tuple[int, int, str]] = []
    for piece, piece_text in sorted(texts.items(), key=lambda item: -len(item[1])):
        if not piece_text:
            continue
        at = text.find(piece_text)
        while at >= 0:
            end = at + len(piece_text)
            if not any(at < e and s < end for s, e, _ in found):
                found.append((at, end, piece))
            at = text.find(piece_text, end)
    return sorted(found)


def soften(messages: list[dict], texts: dict[str, str], block_ids: dict[str, str]) -> tuple[list[dict], set[str]]:
    """Messages with the pieces in `texts` as block parts (`block_ids[piece]`), and the pieces used."""
    used: set[str] = set()
    out = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "assistant" or not isinstance(content, str):
            out.append(message)
            continue
        found = find_pieces(content, texts)
        if not found:
            out.append(message)
            continue
        parts, last = [], 0
        for start, end, piece in found:
            if start > last:
                parts.append({"type": "text", "text": content[last:start]})
            parts.append({"type": "neuralese", "id": block_ids[piece]})
            used.add(piece)
            last = end
        if last < len(content):
            parts.append({"type": "text", "text": content[last:]})
        out.append({**message, "content": parts})
    return out, used


def save_bank(path, bank: PromptBank, rows: dict[str, torch.Tensor], provenance: dict) -> dict[str, str]:
    """Write `rows` (piece ID → trained rows) as a bank; pieces not in `rows` keep the bank's rows. Returns the new
    block IDs by piece."""
    tensors, blocks, values, ids = {}, {}, {}, {}
    for piece in sorted(bank.texts):
        payload = rows.get(piece, bank.rows[piece]).detach().float().cpu().contiguous()
        trained = piece in rows
        block = make_block(payload, bank.dialect, type=TYPE,
                           producer={"kind": "trained" if trained else "text-init", "init_text": bank.texts[piece]})
        tensors[block.id] = block.payload
        blocks[block.id] = {"dialect": bank.dialect, "dtype": "F32", "length": payload.shape[0], "width": payload.shape[1],
                            "producer": block.producer}
        values[piece] = {"$neuralese": {"id": block.id, "type": TYPE}}
        ids[piece] = block.id
    header = {"blocks": blocks, "dialect": bank.dialect, "exports": {EXPORT: {
        "description": "Soft forms of the runtime prompt pieces.", "type": f"Record<string, {TYPE}>", "value": values}},
        "format": "natlang.neuralese-file/1", "provenance": provenance, "types": "type SystemPrompt = string;"}
    save_file(tensors, str(path), metadata={"natlang": json.dumps(header, sort_keys=True, separators=(",", ":"))})
    return ids
