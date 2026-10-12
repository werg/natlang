#!/usr/bin/env python3
"""Create an exact clean direct-stage head anchor for a pinned LFM HF source."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve import load_engine
    from natlang_neuralese.train.direct_export import _file_inventory, lfm_snapshot_source

    base = args.base.resolve(strict=True)
    model_id, revision = lfm_snapshot_source(base, args.hub_cache)
    files = _file_inventory(base)
    receipt_path = args.out.with_suffix(".source.json")
    if args.out.exists() or receipt_path.exists():
        raise FileExistsError(f"clean anchor or source receipt already exists: {args.out}")
    torch.manual_seed(0)
    engine = load_engine(base=str(base), cutoff=8, max_block=256,
                         dtype=torch.bfloat16, device="cpu")
    heads = PortHeads(engine.backbone, cutoff=8, max_length=256,
                      stop_source="shallow", stop_position=True,
                      profile="raw-token-v1")
    heads.set_content_transport("raw-identity")
    state = {
        "port_config": {"cutoff": 8, "max_length": 256, "stop_source": "shallow",
                        "stop_position": True, "profile": "raw-token-v1",
                        "content_transport": "raw-identity"},
        "heads": heads.state_dict(),
        "control_rows": engine.backbone.control_rows.detach().cpu(),
        "control_head_rows": (None if getattr(engine.backbone, "control_head_rows", None) is None else
                              engine.backbone.control_head_rows.detach().cpu()),
        "backbone": {"base": str(base), "model_id": model_id, "revision": revision,
                     "files_sha256": canonical_json_sha256_hex(files)},
        "note": "Fresh QAD-compatible clean direct head anchor; no trained backbone or heads.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending = args.out.with_name(args.out.name + f".pending-{os.getpid()}")
    torch.save(state, pending)
    pending.replace(args.out)
    receipt = {"schema": "natlang.lfm-clean-head-anchor/1", "base": str(base),
               "model_id": model_id, "revision": revision, "files": files,
               "anchor": str(args.out.resolve()), "anchor_sha256": sha256_file_hex(args.out)}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"anchor": str(args.out), "sha256": receipt["anchor_sha256"],
                      "source": model_id, "revision": revision}))


if __name__ == "__main__":
    main()
