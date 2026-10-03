"""Compare cache paths on fixed held-out prefixes and identical committed payloads.

Run against an S3 checkpoint before scaling. This writes diagnostic evidence only;
it never trains, edits the checkpoint, or publishes data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

import torch

from ..data.records import read_records
from ..data.render import Renderer, span_examples
from ..serve import load_engine
from ..write import greedy_continue, open_block, read_back, write_block


@torch.no_grad()
def compare(backbone, heads, prefix, payload, steps):
    device = backbone.embedding_weight.device
    ids = torch.tensor([prefix + [backbone.controls.open_id]], device=device)
    payload = payload.to(device=device, dtype=backbone.embedding_weight.dtype)[None]
    opened = open_block(backbone, heads, ids)
    # Exercise the same snapshot branching as the training harness.
    write_block(backbone, heads, opened)
    back = read_back(backbone, heads, opened.cache, payload)
    control = read_back(backbone, heads, open_block(backbone, heads, ids).cache, payload)
    branch_diff = float((back["logits"].float() - control["logits"].float()).abs().max())
    tokens, cached = greedy_continue(backbone, back["cache"], back["logits"], steps=steps)
    close = torch.tensor([[backbone.controls.close_id]], device=device)
    sequence = torch.cat([backbone.embed(ids), heads.interface(payload), backbone.embed(close),
                          backbone.embed(torch.tensor([tokens], device=device))], 1)
    start = ids.shape[1] + payload.shape[1]
    full = backbone.forward_embeds(sequence)["logits"][0, start:start + steps + 1].float()
    cached = torch.cat(cached, 0).float()
    rows = []
    for i, (c, f) in enumerate(zip(cached, full)):
        ct, ft = c.topk(2), f.topk(2)
        delta = c - f
        rows.append({"step": i, "max_diff": float(delta.abs().max()),
                     "rms_diff": float(delta.square().mean().sqrt()),
                     "cached_token": int(ct.indices[0]), "full_token": int(ft.indices[0]),
                     "cached_margin": float(ct.values[0] - ct.values[1]),
                     "full_margin": float(ft.values[0] - ft.values[1])})
    return {"prefix_tokens": len(prefix), "payload_vectors": payload.shape[1],
            "fork_vs_unforked_logit_diff": branch_diff,
            "first_mismatch_step": next((r["step"] for r in rows[:steps]
                                          if r["cached_token"] != r["full_token"]), None),
            "steps": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", default="qa_extractive,qa_multihop,tool_digest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    engine = load_engine(heads_checkpoint=args.checkpoint, device=args.device)
    backbone, heads = engine.backbone, engine.heads
    renderer = Renderer(engine.tokenizer, backbone.controls)
    paths = [Path(args.records) / f"{f}.port-records.jsonl" for f in args.families.split(",")]
    eval_texts = list(islice((r.source_text() for r in read_records(paths, imitation_only=False)
                             if r.split in ("test", "validation")), 200))
    prefixes = [s.prefix for s in islice(span_examples(renderer, eval_texts, 64, 16, 32, limit=48), 4)]
    if len(prefixes) != 4:
        raise ValueError("Expected the pilot's four held-out prefixes")
    with torch.no_grad():
        payloads = [write_block(backbone, heads, open_block(backbone, heads,
                    torch.tensor([p + [backbone.controls.open_id]], device=args.device))).row(0).float().cpu()
                    for p in prefixes]
    torch.save({"prefixes": prefixes, "payloads": payloads}, out / "fixed-inputs.pt")
    kernel = backbone.conv_kernel
    results = []
    for name, dtype, fast, conv in (
        ("bf16_fast_kernel", torch.bfloat16, True, kernel),
        ("bf16_fast_no_kernel", torch.bfloat16, True, None),
        ("bf16_reference", torch.bfloat16, False, None),
        ("f32_reference", torch.float32, False, None),
    ):
        backbone.hf.to(dtype=dtype)
        # Control rows and trained head parameters retain their trained float32 dtype.
        heads.feedback.embedding = backbone.embedding_weight.detach().clone()
        backbone.fast, backbone.conv_kernel = fast, conv
        cases = [compare(backbone, heads, p, z, args.steps) for p, z in zip(prefixes, payloads)]
        result = {"variant": name, "dtype": str(dtype), "device": args.device,
                  "fast": fast, "conv_kernel_loaded": conv is not None, "cases": cases}
        results.append(result)
        print(json.dumps(result), flush=True)
        (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    # Float32 comparison promotes the same BF16 base weights; it isolates arithmetic,
    # not a separately loaded full-precision checkpoint.
    report = {"checkpoint": args.checkpoint, "input_sha256": hashlib.sha256(
              (out / "fixed-inputs.pt").read_bytes()).hexdigest(),
              "f32_control": "same bf16-loaded base weights promoted to float32", "variants": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
