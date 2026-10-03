"""Throughput of the port's phases: reference path versus fast path.

    python -m natlang_neuralese.eval.bench [--device cuda] [--prefix 2048] [--block 32] [--batch 1]

Reports tokens per second for prefill, the sketch loop (one vector through k layers),
blockwise completion (L vectors through D−k layers) and readback (L+1 positions through all
layers), plus text decoding for comparison. Inference only (no autograd).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ..model.heads import PortHeads  # noqa: E402
from ..model.lfm2_port import ControlTokens, PortBackbone, load_backbone, load_conv_kernel  # noqa: E402
from ..write import greedy_continue, open_block, read_back, write_block  # noqa: E402


def _timed(fn, device, repeats: int):
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    fn()  # warm-up
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    sync()
    return (time.perf_counter() - started) / repeats, result


@torch.no_grad()
def bench(backbone: PortBackbone, heads: PortHeads, prefix: torch.Tensor, block: int, repeats: int) -> dict:
    device = prefix.device
    batch = prefix.shape[0]
    t_prefill, opened = _timed(lambda: open_block(backbone, heads, prefix), device, repeats)
    timings: dict = {}

    def write():
        timings.clear()
        return write_block(backbone, heads, opened, max_length=block, timings=timings)

    _, written = _timed(write, device, repeats)
    t_sketch, t_complete = timings["shallow_generation"], timings["completion"] + timings["projection"]
    t_read, back = _timed(lambda: read_back(backbone, heads, written.block_start, written.payload), device, repeats)
    t_decode, _ = _timed(lambda: greedy_continue(backbone, opened.cache, opened.logits[:1], steps=block)
                         if batch == 1 else None, device, max(1, repeats // 2))
    length = written.payload.shape[1]
    out = {
        "prefill_tok_s": batch * prefix.shape[1] / t_prefill,
        "sketch_vectors_s": batch * length / t_sketch,
        "sketch_ms_per_step": 1e3 * t_sketch / length,
        "completion_vectors_s": batch * length / t_complete,
        "readback_tok_s": batch * (length + 1) / t_read,
        "block_length": length,
    }
    if batch == 1:
        out["text_decode_tok_s"] = block / t_decode
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prefix", type=int, default=2048)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    dtype = getattr(torch, args.dtype) if args.device != "cpu" else torch.float32
    model, tokenizer = load_backbone(dtype=dtype, device="cpu")
    model.to(args.device)
    controls = ControlTokens.from_tokenizer(tokenizer)
    fast = PortBackbone(model, controls, conv_kernel=load_conv_kernel()).to(args.device)
    reference = copy.copy(fast)
    reference.fast = False
    torch.manual_seed(0)
    heads = PortHeads(fast, cutoff=args.cutoff, max_length=args.block).to(args.device).eval()
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 60000, (args.batch, args.prefix - 1), generator=generator)
    prefix = torch.cat([ids, torch.full((args.batch, 1), controls.open_id)], 1).to(args.device)
    report = {"device": args.device, "dtype": str(dtype), "prefix": args.prefix, "block": args.block,
              "batch": args.batch, "cutoff": args.cutoff, "conv_kernel": fast.conv_kernel is not None,
              "reference": bench(reference, heads, prefix, args.block, args.repeats),
              "fast": bench(fast, heads, prefix, args.block, args.repeats)}
    report["speedup"] = {k: report["fast"][k] / report["reference"][k] for k in report["fast"]
                         if k.endswith("_s") and report["reference"].get(k)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
