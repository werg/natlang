"""Nested Maple family training (plans/neuralese/MAPLE_NESTED.md §4, §4a).

One Maple holds the whole family: the full model and members ``LxE`` (first L layers, first E experts per layer, in
the N0 order; depth-nested members exit early through the final norm and the shared head). Two phases:

- ``bootstrap`` (N2a): shared weights frozen; each member's private parts (private router rows, norm gain corrections,
  attention deltas inside the quantizer) are distilled from the full model: logits KL, hidden-state matching at
  shared depths, small CE weight.
- ``joint`` (N2b): everything trains. The full model takes CE + KL to the frozen original Maple (anchor, adapters
  off); every member takes its own CE + KL to the full model + hidden-state matching. Shared parameters: attention
  QAT LoRA and learned block scales, expert block scales (scale-only QAT).

Data mixes the natlang task rows with broad rows (``--mixed``); evaluation reports every member on both.

    python -m natlang_neuralese.maple.nested_train --phase bootstrap --members 24x32,24x64,8x16 \\
        --order runs/maple-nested-20261005/n0-v1/expert-order.pt --data ... --mixed ... --out ...
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..train.joint import load_rows, shifted
from ..train.joint_kd import IGNORE, chunked_ce_kl
from .model import load_maple
from .ternary import adapters_disabled, add_qat_lora, flip_fraction

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class Member:
    key: str
    layers: int | None   # None: all layers
    experts: int

    @staticmethod
    def parse(text: str, total_layers: int) -> "Member":
        layers, _, experts = text.partition("x") if "x" in text else ("", "", text)
        depth = int(layers) if layers else total_layers
        return Member(key=f"{depth}x{int(experts)}", layers=None if depth == total_layers else depth,
                      experts=int(experts))

    def depths(self, total_layers: int) -> list[int]:
        """Residual depths compared with the full model (hidden-state matching): 8, 16 and the exit depth."""
        depth = self.layers or total_layers
        return sorted({d for d in (8, 16) if d < depth} | ({depth} if self.layers else set()))


def setup(model, members: list[Member], rank: int, private_rank: int, learn_scales: bool, expert_scales: bool):
    """Attach the shared QAT parameters and each member's private parts.
    Returns (adapter parameters, scale parameters, private parameters)."""
    adapters, scales, private = [], [], []
    for index, layer in enumerate(model.model.layers):
        for proj in ATTENTION:
            module = getattr(layer.self_attn, proj)
            adapter = add_qat_lora(module, rank=rank, alpha=2.0 * rank)
            adapters += [adapter.lora_A, adapter.lora_B]
            if learn_scales:
                scales.append(adapter.learn_scales(module.parametrizations.weight.original))
            for m in members:
                if m.layers is None or index < m.layers:
                    p = adapter.add_private(m.key, rank=private_rank, alpha=2.0 * private_rank)
                    private += [p.lora_A, p.lora_B]
        if expert_scales:
            scales += layer.mlp.experts.learn_scales()
    for m in members:
        private += model.add_private_router(m.experts, key=m.key, layers=m.layers)
        private += model.add_private_norms(m.key, layers=m.layers)
    return adapters, scales, private


def hidden_match(student: dict, teacher: dict, depths) -> torch.Tensor:
    """Mean over depths of the normalised squared distance ‖h_s − h_t‖² / ‖h_t‖² (teacher detached)."""
    terms = []
    for d in depths:
        t = teacher[d].detach().float()
        s = student[d].float()
        terms.append((s - t).pow(2).sum(-1).mean() / t.pow(2).sum(-1).mean().clamp_min(1e-6))
    if not terms:
        return torch.zeros((), device=next(iter(teacher.values())).device) if teacher else torch.zeros(())
    return torch.stack(terms).mean()


def run(model, member: Member | None, x, capture=()):
    """Forward as ``member`` (None: the full model); always restores the full model afterwards."""
    if member is None:
        model.set_member(None)
    else:
        model.set_member(member.key, experts=member.experts, layers=member.layers)
    try:
        return model.model(input_ids=x, capture=capture)
    finally:
        model.set_member(None)


def member_step(model, members, x, y, *, phase, normaliser, ce_member, kl_weight, hidden_weight, anchor_weight,
                chunk, total_layers, scale=1.0):
    """One micro-step on one sequence: the full model (joint phase) and each member; calls backward. Returns sums."""
    head = model.get_output_embeddings().weight
    sums: dict[str, float] = {}

    def add(name, value):
        sums[name] = sums.get(name, 0.0) + float(value)

    depths = sorted({d for m in members for d in m.depths(total_layers)})
    if phase == "joint":
        # The anchor (KL to the frozen original Maple) only guards general ability on broad rows: callers pass
        # anchor_weight=0 for task rows, where the full model is meant to change.
        original = None
        if anchor_weight:
            with torch.no_grad(), adapters_disabled():
                original = run(model, None, x).last_hidden_state
        full_out = run(model, None, x, capture=depths)
        loss, p = chunked_ce_kl(full_out.last_hidden_state, head, y, original, head if anchor_weight else None,
                                kl_weight=anchor_weight, normaliser=normaliser, chunk=chunk)
        (loss * scale).backward()
        add("full_ce", p.ce)
        add("anchor_kl", p.kl)
        full_h = full_out.last_hidden_state.detach()
        full_cap = {d: h.detach() for d, h in full_out.captured.items()}
        del full_out, loss
    else:
        with torch.no_grad():
            full_out = run(model, None, x, capture=depths)
        full_h, full_cap = full_out.last_hidden_state, full_out.captured
    for m in members:
        out_m = run(model, m, x, capture=m.depths(total_layers))
        lm, p = chunked_ce_kl(out_m.last_hidden_state, head, y, full_h, head, ce_weight=ce_member,
                              kl_weight=kl_weight, normaliser=normaliser * len(members), chunk=chunk)
        hm = hidden_match(out_m.captured, full_cap, m.depths(total_layers))
        ((lm + hidden_weight * hm * p.tokens / (normaliser * len(members))) * scale).backward()
        add(f"{m.key}/ce", p.ce)
        add(f"{m.key}/kl", p.kl)
        add(f"{m.key}/hidden", hm.item() * p.tokens)
    add("tokens", p.tokens)
    return sums


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="/home/werg/data/models/maple-preview-converted")
    ap.add_argument("--order", help="expert-order.pt from natlang_neuralese.maple.routing (N0)")
    ap.add_argument("--members", default="24x32,24x64,8x16")
    ap.add_argument("--phase", choices=["bootstrap", "joint"], default="bootstrap")
    ap.add_argument("--resume", help="trainable state from a previous phase (nested-state.pt)")
    ap.add_argument("--data", required=True, help="natlang task rows")
    ap.add_argument("--sample-rows", type=int, default=0,
                    help="task rows to sample and tokenize (0: as many as the run consumes; -1: all)")
    ap.add_argument("--mixed", help="broad rows (scripts/maple_mixed_data.py)")
    ap.add_argument("--mixed-fraction", type=float, default=0.0,
                    help="share of broad rows (owner: forgetting general ability is acceptable; default off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--members-per-step", type=int, default=0, help="0: every member every step")
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--private-rank", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--private-lr", type=float, default=1e-3)
    ap.add_argument("--scale-lr", type=float, default=1e-5)
    ap.add_argument("--no-learned-scales", action="store_true")
    ap.add_argument("--no-expert-scales", action="store_true")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--hidden-weight", type=float, default=1.0)
    ap.add_argument("--anchor-weight", type=float, default=0.0,
                    help="KL to the frozen original Maple on broad rows only (0: off); never on task rows")
    ap.add_argument("--member-ce-weight", type=float, default=None, help="default 0.1 bootstrap, 1.0 joint")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--eval-rows", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ce_member = args.member_ce_weight if args.member_ce_weight is not None else (0.1 if args.phase == "bootstrap" else 1.0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)  # Qwen2Tokenizer; never remote code
    needed = args.steps * args.accumulate + args.eval_rows
    task = load_rows(args.data, tokenizer, args.max_length, "train",
                     sample=args.sample_rows or needed if args.sample_rows != -1 else None, seed=args.seed)
    random.shuffle(task)
    mixed = load_rows(args.mixed, tokenizer, args.max_length, "train") if args.mixed else []
    random.shuffle(mixed)
    held = {"task": task[:args.eval_rows], "mixed": mixed[:args.eval_rows]}
    task, mixed = task[args.eval_rows:], mixed[args.eval_rows:]

    start = time.time()
    model = load_maple(args.model, device="cuda", ternary_attention=False, cache=args.cache or None)
    if args.order:
        model.order_experts(torch.load(args.order)["orders"])
    total_layers = model.config.num_hidden_layers
    members = [Member.parse(m, total_layers) for m in args.members.split(",")]
    adapters, scales, private = setup(model, members, args.rank, args.private_rank, not args.no_learned_scales,
                                      not args.no_expert_scales)
    if args.resume:
        state = torch.load(args.resume, map_location="cuda")
        result = model.load_state_dict(state["trainable"], strict=False)
        print(json.dumps({"event": "resumed", "from": args.resume, "unexpected": len(result.unexpected_keys)}))
    groups = [{"params": private, "lr": args.private_lr}]
    if args.phase == "joint":
        groups += [{"params": adapters, "lr": args.lr}, {"params": scales, "lr": args.scale_lr}]
    else:
        for p in adapters + scales:
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    trainable = [p for g in groups for p in g["params"]]
    log = open(out / "train.jsonl", "a", buffering=1)
    head = model.get_output_embeddings().weight

    def emit(record):
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")

    def evaluate():
        result = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for split, rows in held.items():
                for ids, labels in rows:
                    x, y = shifted(ids, labels, "cuda")
                    with adapters_disabled():
                        original = run(model, None, x).last_hidden_state
                    full = run(model, None, x).last_hidden_state
                    _, p = chunked_ce_kl(full, head, y, original, head, kl_weight=1.0, chunk=args.chunk)
                    r = result.setdefault(f"{split}/full", [0.0, 0.0, 0])
                    r[0] += p.ce; r[1] += p.kl; r[2] += p.tokens
                    for m in members:
                        h = run(model, m, x).last_hidden_state
                        _, p = chunked_ce_kl(h, head, y, full, head, kl_weight=1.0, chunk=args.chunk)
                        r = result.setdefault(f"{split}/{m.key}", [0.0, 0.0, 0])
                        r[0] += p.ce; r[1] += p.kl; r[2] += p.tokens
        # full: kl to the original Maple; members: kl to the current full model
        return {k: {"ce": round(c / max(n, 1), 4), "kl": round(kl / max(n, 1), 4)} for k, (c, kl, n) in result.items()}

    def save(step):
        names = {id(p) for p in trainable}
        torch.save({"trainable": {k: v.detach().cpu() for k, v in model.state_dict(keep_vars=True).items()
                                  if id(v) in names},
                    "members": [m.key for m in members], "order": args.order, "phase": args.phase, "step": step,
                    "model": args.model, "args": vars(args)}, out / "nested-state.pt")

    emit({"event": "start", "phase": args.phase, "members": [m.key for m in members], "args": vars(args),
          "load_seconds": round(time.time() - start, 1), "adapters": sum(p.numel() for p in adapters),
          "scales": sum(p.numel() for p in scales), "private": sum(p.numel() for p in private),
          "gpu_gb": round(torch.cuda.memory_allocated() / 2**30, 1), "eval": evaluate()})
    torch.cuda.reset_peak_memory_stats()
    task_cursor = mixed_cursor = member_cursor = 0
    for step in range(1, args.steps + 1):
        began = time.time()
        batch = []
        for _ in range(args.accumulate):
            if mixed and random.random() < args.mixed_fraction:
                batch.append(("mixed", mixed[mixed_cursor % len(mixed)]))
                mixed_cursor += 1
            else:
                batch.append(("task", task[task_cursor % len(task)]))
                task_cursor += 1
        normaliser = sum(sum(1 for t in labels[1:] if t != IGNORE) for _, (_, labels) in batch) or 1
        k = args.members_per_step or len(members)
        chosen = [members[(member_cursor + i) % len(members)] for i in range(k)]
        member_cursor += k
        sums: dict[str, float] = {}
        for kind, (ids, labels) in batch:
            x, y = shifted(ids, labels, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                part = member_step(model, chosen, x, y, phase=args.phase, normaliser=normaliser, ce_member=ce_member,
                                   kl_weight=args.kl_weight, hidden_weight=args.hidden_weight,
                                   anchor_weight=args.anchor_weight if kind == "mixed" else 0.0, chunk=args.chunk,
                                   total_layers=total_layers)
            for name, value in part.items():
                sums[f"{kind}/{name}"] = sums.get(f"{kind}/{name}", 0.0) + value
            sums["tokens"] = sums.get("tokens", 0.0) + part.get("tokens", 0.0)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        sums.pop("tokens")
        per_kind = {kind: max(sums.pop(f"{kind}/tokens", 0.0), 1) for kind in ("task", "mixed")}
        record = {"step": step, **{name: round(v / per_kind[name.split("/")[0]], 4) for name, v in sums.items()},
                  "members": [m.key for m in chosen], "grad_norm": round(norm, 3),
                  "seconds": round(time.time() - began, 2),
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if args.phase == "joint" and (step % 10 == 0 or step == args.steps):
            flips = [flip_fraction(mod) for name, mod in model.named_modules()
                     if name.endswith(("q_proj", "o_proj")) and hasattr(mod, "parametrizations")]
            record["flip_fraction_mean"] = round(sum(flips) / len(flips), 6)
        emit(record)
        if step % args.eval_every == 0 or step == args.steps:
            emit({"event": "eval", "step": step, "eval": evaluate()})
            save(step)


if __name__ == "__main__":
    main()
