"""N0/N1 of plans/neuralese/MAPLE_NESTED.md: expert routing statistics of a full Maple-family model (Maple, Mellum) on
our corpus, the per-layer expert order, and what nested prefixes and LxE members (``--members``) cost before any
training.

    python -m natlang_neuralese.maple.routing --model MODEL_DIR \\
        --data runs/maple-joint-20261005/qwen3-render-v1.jsonl --out runs/maple-nested-20261005/n0-v1

Rows are rendered SFT turns (``prompt`` + ``completion``). Rows are split by program into an ordering part
(statistics are collected over every token of these rows) and a held-out part, where completion-token negative
log-likelihood and next-token agreement with the full model are measured for each prefix size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ..common.paths import resolve_str
from ..train.joint_kd import IGNORE, chunked_ce_kl
from .model import load_maple

SIZES = (16, 24, 32, 40, 48, 64, 96, 128, 256)


def held_out(row, fraction):
    key = row.get("program_id") or row["id"]
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32 < fraction


def coverage(mass: torch.Tensor, sizes) -> dict:
    ordered = mass.sort(descending=True).values
    total = ordered.sum().item() or 1.0
    return {str(n): round(ordered[:n].sum().item() / total, 4) for n in sizes if n <= mass.numel()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="default",
                    help="converted-model cache (read if present, written otherwise); empty to disable")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--held-out", type=float, default=0.15)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--members", default="",
                    help="family members LxE (first L layers, first E experts) also measured after the ordering, "
                         "e.g. 28x16,28x24,28x32,14x16 for Mellum: depth members need their own numbers")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)  # Qwen2Tokenizer; never the remote code
    rows = [json.loads(line) for line in open(args.data)]
    if args.max_rows:
        rows = rows[:args.max_rows]
    start = time.time()
    from .student import default_cache

    cache = default_cache(args.model) if args.cache == "default" else (args.cache or None)
    model = load_maple(args.model, device=args.device, cache=cache)
    print(json.dumps({"event": "loaded", "seconds": round(time.time() - start, 1),
                      "gpu_gb": round(torch.cuda.memory_allocated() / 2**30, 1) if args.device == "cuda" else None}),
          flush=True)

    def encode(row):
        prompt = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        ids = (prompt + completion)[-args.max_length:]
        supervised = min(len(completion), len(ids))
        labels = [IGNORE] * (len(ids) - supervised) + ids[len(ids) - supervised:]
        return torch.tensor([ids], device=args.device), torch.tensor([labels[1:] + [IGNORE]], device=args.device)

    order_rows = [r for r in rows if not held_out(r, args.held_out)]
    test_rows = [r for r in rows if held_out(r, args.held_out)]

    # N0: routing statistics on the ordering rows, full model
    stats = model.collect_routing()
    with torch.no_grad():
        for i, row in enumerate(order_rows):
            ids, _ = encode(row)
            model.model(input_ids=ids)
            if i % 25 == 0:
                print(json.dumps({"event": "statistics", "row": i, "of": len(order_rows),
                                  "seconds": round(time.time() - start, 1)}), flush=True)
    model.collect_routing(False)
    layers = []
    for index, s in enumerate(stats):
        layers.append({"layer": index, "tokens": s["tokens"], "coverage_by_mass": coverage(s["mass"], sizes),
                       "coverage_by_count": coverage(s["count"], sizes),
                       "unused_experts": int((s["count"] == 0).sum())})
    orders = [torch.argsort(s["mass"], descending=True) for s in stats]
    torch.save({"orders": orders, "mass": [s["mass"] for s in stats], "count": [s["count"] for s in stats]},
               out / "expert-order.pt")
    model.order_experts(orders)

    # N1 (untrained): held-out completion NLL and agreement with the full model for each prefix size
    head = model.get_output_embeddings().weight
    results = {n: {"nll": 0.0, "kl_from_full": 0.0, "agree": 0, "tokens": 0} for n in sizes}
    with torch.no_grad():
        for row in test_rows:
            ids, labels = encode(row)
            model.set_active_experts(None)
            full = model.model(input_ids=ids).last_hidden_state
            mask = labels[0] != IGNORE
            full_top = (full[0][mask].float() @ head.float().T).argmax(-1)
            for n in sizes:
                model.set_active_experts(None if n >= model.config.num_experts else n)
                h = full if n >= model.config.num_experts else model.model(input_ids=ids).last_hidden_state
                _, parts = chunked_ce_kl(h, head, labels, full, head, kl_weight=1.0)
                top = (h[0][mask].float() @ head.float().T).argmax(-1)
                r = results[n]
                r["nll"] += parts.ce
                r["kl_from_full"] += parts.kl
                r["agree"] += int((top == full_top).sum())
                r["tokens"] += parts.tokens
    model.set_active_experts(None)
    members = [m for m in args.members.split(",") if m]
    for spec in members:  # depth x width members (early exit after L layers, first E experts per layer)
        depth, width = (int(v) for v in spec.lower().split("x"))
        results[spec] = r = {"nll": 0.0, "kl_from_full": 0.0, "agree": 0, "tokens": 0}
        with torch.no_grad():
            for row in test_rows:
                ids, labels = encode(row)
                model.set_member(None)
                full = model.model(input_ids=ids).last_hidden_state
                mask = labels[0] != IGNORE
                full_top = (full[0][mask].float() @ head.float().T).argmax(-1)
                model.set_member(spec, experts=width, layers=depth)
                h = model.model(input_ids=ids).last_hidden_state
                _, parts = chunked_ce_kl(h, head, labels, full, head, kl_weight=1.0)
                r["nll"] += parts.ce
                r["kl_from_full"] += parts.kl
                r["agree"] += int(((h[0][mask].float() @ head.float().T).argmax(-1) == full_top).sum())
                r["tokens"] += parts.tokens
        model.set_member(None)
    nested = {str(n): {"nll": round(r["nll"] / max(r["tokens"], 1), 4),
                       "kl_from_full": round(r["kl_from_full"] / max(r["tokens"], 1), 4),
                       "top1_agreement": round(r["agree"] / max(r["tokens"], 1), 4), "tokens": r["tokens"]}
              for n, r in results.items()}
    report = {"schema": "natlang.maple_nested_n0/1", "model": args.model, "data": args.data,
              "order_rows": len(order_rows), "held_out_rows": len(test_rows), "max_length": args.max_length,
              "layers": layers, "nested_untrained": nested, "seconds": round(time.time() - start, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"event": "done", "nested_untrained": nested}), flush=True)


if __name__ == "__main__":
    main()
