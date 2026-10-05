"""Nested Maple training (plans/neuralese/MAPLE_NESTED.md §4): one Maple, trained as the full model and as its
expert prefixes at once, with shared ternary QAT LoRA on attention and a private router bias per prefix size.

    python -m natlang_neuralese.maple.nested_train --model /home/werg/data/models/maple-preview-bf16 \\
        --order runs/maple-nested-20261005/n0-v1/expert-order.pt --sizes 32:0.6,64:0.2,128:0.2 \\
        --data runs/maple-joint-20261005/qwen3-render-v1.jsonl --out runs/maple-nested-20261005/n2-smoke

Per micro-step: the full model takes its cross-entropy; a prefix size n is sampled and the nested model takes its
cross-entropy plus KL to the full model's detached distribution. The frozen weights are shared, so the nested model
costs activations only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from ..train.joint import load_rows, shifted
from ..train.joint_kd import chunked_ce_kl, kl_ramp
from .model import load_maple
from .ternary import add_qat_lora, qat_adapters

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_sizes(text: str) -> list[tuple[int, float]]:
    sizes = []
    for part in text.split(","):
        n, _, w = part.partition(":")
        sizes.append((int(n), float(w or 1.0)))
    return sizes


def prepare(model, sizes, rank: int, alpha: float) -> list[torch.nn.Parameter]:
    """Attach the shared QAT LoRA (attention) and the per-size router biases; return the trainable parameters."""
    for layer in model.model.layers:
        for proj in ATTENTION:
            add_qat_lora(getattr(layer.self_attn, proj), rank=rank, alpha=alpha)
    params = [p for a in qat_adapters(model).values() for p in a.parameters()]
    for n, _ in sizes:
        if n < model.config.num_experts:
            params += model.add_size_bias(n)
    return params


def nested_losses(model, ids, labels, n, *, kl_weight, small_weight, normaliser, chunk):
    """Full CE + small_weight * (nested CE + kl_weight * KL(full || nested)). Returns (loss, parts dict)."""
    head = model.get_output_embeddings().weight
    model.set_active_experts(None)
    full = model.model(input_ids=ids).last_hidden_state
    loss_full, pf = chunked_ce_kl(full, head, labels, normaliser=normaliser, chunk=chunk)
    model.set_active_experts(n)
    try:
        small = model.model(input_ids=ids).last_hidden_state
    finally:
        model.set_active_experts(None)
    loss_small, ps = chunked_ce_kl(small, head, labels, full.detach(), head, kl_weight=kl_weight,
                                   normaliser=normaliser, chunk=chunk)
    return loss_full + small_weight * loss_small, {"full_ce": pf.ce, "small_ce": ps.ce, "kl": ps.kl,
                                                    "tokens": pf.tokens}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--order", help="expert-order.pt from natlang_neuralese.maple.routing (N0)")
    ap.add_argument("--sizes", default="32:0.6,64:0.2,128:0.2")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-ramp", type=int, default=0)
    ap.add_argument("--small-weight", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--eval-rows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sizes = parse_sizes(args.sizes)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_rows(args.data, tokenizer, args.max_length, "train")
    random.shuffle(rows)
    held, train = rows[:args.eval_rows], rows[args.eval_rows:]
    start = time.time()
    model = load_maple(args.model, device="cuda", ternary_attention=False)
    if args.order:
        model.order_experts(torch.load(args.order)["orders"])
    params = prepare(model, sizes, args.rank, 2.0 * args.rank)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    log = open(out / "train.jsonl", "a", buffering=1)

    def emit(record):
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")

    def evaluate():
        head = model.get_output_embeddings().weight
        totals = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ids, labels in held:
                x, y = shifted(ids, labels, "cuda")
                model.set_active_experts(None)
                full = model.model(input_ids=x).last_hidden_state
                for n, _ in [(model.config.num_experts, 1.0)] + sizes:
                    if n < model.config.num_experts:
                        model.set_active_experts(n)
                        h = model.model(input_ids=x).last_hidden_state
                    else:
                        h = full
                    _, p = chunked_ce_kl(h, head, y, full, head, kl_weight=1.0, chunk=args.chunk)
                    t = totals.setdefault(str(n), [0.0, 0.0, 0])
                    t[0] += p.ce
                    t[1] += p.kl
                    t[2] += p.tokens
                model.set_active_experts(None)
        return {n: {"ce": round(c / max(k, 1), 4), "kl_from_full": round(kl / max(k, 1), 4)}
                for n, (c, kl, k) in totals.items()}

    emit({"event": "start", "args": vars(args), "load_seconds": round(time.time() - start, 1),
          "trainable": sum(p.numel() for p in params),
          "gpu_gb": round(torch.cuda.memory_allocated() / 2**30, 1), "eval": evaluate()})
    torch.cuda.reset_peak_memory_stats()
    cursor = 0
    names, weights = [n for n, _ in sizes], [w for _, w in sizes]
    for step in range(1, args.steps + 1):
        began = time.time()
        batch = [train[(cursor + i) % len(train)] for i in range(args.accumulate)]
        cursor += args.accumulate
        normaliser = sum(sum(1 for t in labels[1:] if t != -100) for _, labels in batch) or 1
        kl = kl_ramp(step - 1, args.kl_ramp, args.kl_weight)
        sums = {"full_ce": 0.0, "small_ce": 0.0, "kl": 0.0, "tokens": 0}
        chosen = []
        for ids, labels in batch:
            n = random.choices(names, weights)[0]
            n = None if n >= model.config.num_experts else n
            chosen.append(n)
            x, y = shifted(ids, labels, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if n is None:
                    loss, parts = nested_losses(model, x, y, None, kl_weight=0.0, small_weight=0.0,
                                                normaliser=normaliser, chunk=args.chunk)
                else:
                    loss, parts = nested_losses(model, x, y, n, kl_weight=kl, small_weight=args.small_weight,
                                                normaliser=normaliser, chunk=args.chunk)
            loss.backward()
            for k in sums:
                sums[k] += parts[k]
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        k = max(sums.pop("tokens"), 1)
        emit({"step": step, **{name: round(v / k, 4) for name, v in sums.items()}, "sizes": chosen,
              "grad_norm": round(norm, 3), "seconds": round(time.time() - began, 2),
              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
    emit({"event": "end", "eval": evaluate()})
    torch.save({"adapters": {n: a.state_dict() for n, a in qat_adapters(model).items()},
                "size_bias": {f"{i}.{k}": v.detach().cpu() for i, layer in enumerate(model.model.layers)
                              for k, v in layer.mlp.size_bias.items()},
                "order": args.order, "sizes": sizes, "model": args.model}, out / "nested-adapters.pt")


if __name__ == "__main__":
    main()
