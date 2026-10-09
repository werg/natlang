"""The combinator standard library in Python (S5 §4.1–4.2; ts-host/src/neuralese/combinators.ts).

The combinator bodies (`map`, `zip`, `ap`, `combine`, `split`, `splitList`, `read`, `convert`, `gloss`) are soft
values in a `.nz` file. `build_text_library` initialises every body from its text description on any backbone's
engine, the same way `buildStandardLibrary` does on a server: encoded in one pass through the port (decision 42), or
the description's token embeddings (`method="embed"`). The descriptions and runtime types are read from the TS
source, which stays their single definition. The file loads with `loadStandardLibrary` and trains as values in
`train.graph_replay`; `rewrite_nz` writes a library (or any `.nz`) with trained bodies in place of the old ones.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .serve.store import make_block

COMBINATORS_TS = Path(__file__).resolve().parents[3] / "ts-host" / "src" / "neuralese" / "combinators.ts"
NAMES = ("map", "zip", "ap", "combine", "split", "splitList", "read", "convert", "gloss")
FORMAT = "natlang.neuralese-file/1"


def combinator_texts(source: Path = COMBINATORS_TS) -> dict[str, dict[str, str]]:
    """name → {type, text} from the `COMBINATORS` table of the TS library."""
    text = source.read_text()
    table = text[text.index("export const COMBINATORS = {"):text.index("} as const;")]
    found = {}
    for match in re.finditer(r"(\w+): \{ type: '([^']*)',\s*text: '([^']*)' \}", table):
        found[match.group(1)] = {"type": match.group(2), "text": match.group(3)}
    missing = [name for name in NAMES if name not in found]
    if missing:
        raise ValueError(f"{source}: no description for {missing}")
    return {name: found[name] for name in NAMES}


def _nz_block_header(block) -> dict:
    return {"dialect": block.dialect, "dtype": "F32", "length": block.length, "width": block.width,
            **({"producer": block.producer} if block.producer else {})}


def build_text_library(engine, path, *, method: str = "encode", provenance: dict | None = None) -> dict[str, str]:
    """Write the text-initialised standard library for `engine`'s backbone to `path`. Returns name → body block ID."""
    from .serve.grad import embed_text, encode_text

    tensors, blocks, exports, bodies = {}, {}, {}, {}
    for name, entry in combinator_texts().items():
        kind = f"Neuralese<{entry['type']}>"
        block = (encode_text if method == "encode" else embed_text)(engine, entry["text"], kind)
        tensors[block.id] = block.payload.detach().float().cpu().contiguous()
        blocks[block.id] = _nz_block_header(block)
        exports[name] = {"type": kind, "description": entry["text"],
                         "value": {"$neuralese-fn": {"type": kind, "body": block.id, "captures": {}}}}
        bodies[name] = block.id
    header = {"format": FORMAT, "dialect": engine.dialect, "exports": exports, "blocks": blocks,
              "provenance": {"kind": "natlang-standard-library",
                             "initialisation": "text-encode" if method == "encode" else "text-embeddings",
                             **(provenance or {})}}
    save_file(tensors, str(path), metadata={"natlang": json.dumps(header, sort_keys=True, separators=(",", ":"))})
    return bodies


def read_blocks(path) -> tuple[dict, dict[str, torch.Tensor]]:
    """(header, block ID → [length, width] float32) of a `.nz` file."""
    with safe_open(str(path), framework="pt") as stream:
        header = json.loads(stream.metadata()["natlang"])
        return header, {key: stream.get_tensor(key).float() for key in stream.keys()}


def library_bodies(path) -> dict[str, str]:
    """name → body block ID of the soft-function exports in a library file."""
    header, _ = read_blocks(path)
    out = {}
    for name, entry in header["exports"].items():
        value = entry.get("value")
        if isinstance(value, dict) and "$neuralese-fn" in value:
            out[name] = value["$neuralese-fn"]["body"]
    return out


def load_into_store(engine, path) -> dict[str, str]:
    """Register every block of a `.nz` file in the engine's store; returns old ID → stored ID (equal unless the
    file's IDs were computed under another rule)."""
    header, tensors = read_blocks(path)
    ids = {}
    for block_id, meta in header["blocks"].items():
        block = engine.store.put(make_block(tensors[block_id], meta["dialect"], type=None,
                                            producer=meta.get("producer")))
        ids[block_id] = block.id
    return ids


def rewrite_nz(source, target, replaced: dict[str, torch.Tensor], *, provenance: dict | None = None) -> dict[str, str]:
    """Copy `source` to `target` with the blocks in `replaced` (old ID → new payload) swapped for new blocks, every
    reference to an old ID pointing at its new block. A replaced block's producer records what it was trained from.
    Returns old ID → new ID."""
    header, tensors = read_blocks(source)
    mapping = {}
    for old, payload in replaced.items():
        meta = header["blocks"][old]
        producer = dict(meta.get("producer") or {})
        init_text = producer.get("init_text", producer.get("text"))
        new_producer = {"kind": "trained", "from": old, **({"init_text": init_text} if init_text else {})}
        block = make_block(payload.detach().float().cpu(), meta["dialect"], producer=new_producer)
        mapping[old] = block.id
        del tensors[old], header["blocks"][old]
        tensors[block.id] = block.payload.contiguous()
        header["blocks"][block.id] = _nz_block_header(block)
    exports = json.dumps(header["exports"])
    for old, new in mapping.items():
        exports = exports.replace(old, new)
    header["exports"] = json.loads(exports)
    if provenance:
        header["provenance"] = {**header.get("provenance", {}), **provenance}
    save_file(tensors, str(target), metadata={"natlang": json.dumps(header, sort_keys=True, separators=(",", ":"))})
    return mapping
