"""Full-latent QAT conversion of a BF16 Maple-family model (Mellum) to ternary (plans/mellum-port.md, owner 2026-10-09).

Every attention projection and every expert matrix becomes a BF16 latent whose forward value is its ternary codes
(Maple's per-row rule: threshold 0.7 x row absmean, scale = mean kept magnitude; straight-through gradient). Router,
norms, embedding and head stay floating, as in Maple's deployed form. Recovery: CE on our corpus plus a sparse KL to
the BF16 original's top-k next-token distribution, dumped once beforehand (``teacher`` phase), so the original never
has to sit next to the student. The KL to the original belongs to this conversion only (owner: no anchor may cap
learning afterwards).

    python -m natlang_neuralese.maple.qat_convert teacher --model DIR --text-data T --out DIR [--windows 512]
    python -m natlang_neuralese.maple.qat_convert train --model DIR --teacher DIR --out DIR [--steps 2000]
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize

from .model import DenseExperts, load_maple
from .ternary import QUANT_MIX, ramped_ternarize_ste, ternarize_ste

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


class TernarySTE(nn.Module):
    """Parametrization: the latent's ternary codes times its row scale, straight-through to the latent."""

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return ramped_ternarize_ste(latent)


def install_full_latent_qat(model) -> list[tuple[str, nn.Parameter]]:
    """Make every attention projection and expert matrix a trainable latent with a ternary forward. Returns the
    latents (name, parameter); everything else is frozen."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    latents = []
    for index, layer in enumerate(model.model.layers):
        for proj in ATTENTION:
            module = getattr(layer.self_attn, proj)
            if not parametrize.is_parametrized(module, "weight"):
                parametrize.register_parametrization(module, "weight", TernarySTE())
            latent = module.parametrizations.weight.original
            latent.requires_grad_(True)
            latents.append((f"model.layers.{index}.self_attn.{proj}.weight", latent))
        experts = layer.mlp.experts
        if not isinstance(experts, DenseExperts):
            raise ValueError("full-latent QAT needs BF16 (dense) experts")
        for name, latent in zip(("gate_up", "down"), experts.make_latent()):
            latents.append((f"model.layers.{index}.mlp.experts.{name}", latent))
    return latents


@torch.no_grad()
def ternary_scale(latent: torch.Tensor) -> float:
    """Mean kept magnitude of the latent's ternary codes: the unit in which its learning rate is set."""
    value = ternarize_ste(latent.detach()).float()
    nonzero = value != 0
    return float(value.abs().sum() / nonzero.sum().clamp_min(1))


@torch.no_grad()
def row_scale(latent: torch.Tensor) -> torch.Tensor:
    """Per-row ternary scale (Maple's rule), shape (..., 1), FP32."""
    from .ternary import ternary_codes

    return ternary_codes(latent.detach())[1].float()


def windows_from(text_data: str, tokenizer, tokens: int, count: int, split: str | None = None,
                 skip_system: bool = False, extra_files: tuple[str, ...] = ()) -> list[list[int]]:
    """Token windows packed across documents. ``skip_system`` drops each chat's system message (our shared NatLang
    prompt: ~86% of corpus tokens, not predicted by chat models, memorised in one update), so recovery runs on
    the turns the teacher actually models. ``extra_files`` (code, prose) are packed in alternately."""
    def documents():
        for line in open(text_data):
            row = json.loads(line)
            if split and row.get("split", "train") != split:
                continue
            text = row["text"]
            if skip_system and text.startswith("<|im_start|>system") and "<|im_end|>" in text:
                text = text[text.index("<|im_end|>") + len("<|im_end|>"):].lstrip("\n")
            yield text

    def extras():
        for path in extra_files:
            yield Path(path).read_text(errors="replace")

    rows, buffer = [], []
    sources = [documents()] + ([extras()] if extra_files else [])
    turn = 0
    while sources and len(rows) < count:
        source = sources[turn % len(sources)]
        try:
            text = next(source)
        except StopIteration:
            sources.remove(source)
            continue
        turn += 1
        buffer += tokenizer(text, add_special_tokens=False).input_ids
        while len(buffer) >= tokens and len(rows) < count:
            rows.append(buffer[:tokens])
            buffer = buffer[tokens:]
    return rows


def topk_kl(student_logits: torch.Tensor, teacher_ids: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over the teacher's top-k (its probabilities renormalised over those k)."""
    t = teacher_logp.float()
    t = t - torch.logsumexp(t, -1, keepdim=True)
    s = torch.log_softmax(student_logits.float(), -1).gather(-1, teacher_ids.long())
    return (t.exp() * (t - s)).sum(-1).mean()


def hidden_logits(model, ids):
    return model(ids).logits


def run_teacher(a):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = load_maple(a.model, device="cuda", dtype=torch.bfloat16, ternary_attention=False).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for split, count in (("train", a.windows), ("test", a.held)):
        extra = tuple(sorted(str(f) for g in a.extra_glob for f in Path("/").glob(g.lstrip("/"))))
        extra = extra[0::2] if split == "train" else extra[1::2]  # disjoint, interleaved extra files
        windows = windows_from(a.text_data, tokenizer, a.tokens, count, split, skip_system=a.skip_system,
                               extra_files=extra)
        ids_all, top_ids, top_logp, ce = [], [], [], []
        with torch.no_grad():
            for w in windows:
                ids = torch.tensor([w], device="cuda")
                logits = model(ids).logits[0, :-1].float()
                logp = torch.log_softmax(logits, -1)
                values, index = logp.topk(a.k, -1)
                ce.append(float(-logp.gather(-1, ids[0, 1:, None]).mean()))
                ids_all.append(ids[0].cpu().int())
                top_ids.append(index.cpu().int())
                top_logp.append(values.cpu().half())
        torch.save({"ids": torch.stack(ids_all), "top_ids": torch.stack(top_ids), "top_logp": torch.stack(top_logp),
                    "teacher_ce": ce}, out / f"{split}.pt")
        print(json.dumps({"split": split, "windows": len(windows), "teacher_ce": sum(ce) / max(1, len(ce))}), flush=True)
    (out / "teacher.json").write_text(json.dumps({"model": a.model, "text_data": a.text_data, "tokens": a.tokens,
                                                  "k": a.k, "skip_system": a.skip_system,
                                                  "extra_glob": a.extra_glob}, indent=1) + "\n")


def run_train(a):
    from ..train.optim import LionSR

    teacher_dir = Path(a.teacher)
    train, held = torch.load(teacher_dir / "train.pt"), torch.load(teacher_dir / "test.pt")
    model = load_maple(a.model, device="cuda", dtype=torch.bfloat16, ternary_attention=False)
    latents = install_full_latent_qat(model)
    model.model.checkpoint_layers = a.checkpoint_layers
    scales = {name: ternary_scale(latent) for name, latent in latents}
    # Step per row in units of that row's ternary scale: a per-tensor unit moved near-zero rows (layers 0-3, 27
    # gate_up) by up to 3.6x their RMS per update.
    groups = [{"params": [latent], "lr": a.lr, "row_scale": row_scale(latent), "name": name} for name, latent in latents]
    optimizer = LionSR(groups, lr=a.lr, fused=True)
    QUANT_MIX["fused"] = True
    optimizer.step_in_backward()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from ..train.checkpoint_policy import CheckpointPolicy

    policy = CheckpointPolicy(out, every_minutes=a.checkpoint_minutes).install_signal_handlers()
    state_path = policy.slot
    step = 0
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", mmap=True)  # paged from the file, not a second copy
        with torch.no_grad():
            for name, latent in latents:
                latent.copy_(state["latents"][name].to(latent))
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
    log = (out / "train.jsonl").open("a")

    def schedule(at):
        """Ternarization mix and BF16-teacher KL weight at update ``at``: the mix ramps to 1 (deployed), the KL to
        the original phases out (owner: no anchor to the original model beyond the conversion)."""
        mix = min(1.0, at / a.ramp_steps) if a.ramp_steps else 1.0
        kl = a.kl_weight * max(0.0, 1.0 - at / a.kl_decay_steps) if a.kl_decay_steps else a.kl_weight
        return mix, kl

    @torch.no_grad()
    def evaluate():
        """Held CE/KL of the deployed model (mix 1), and of the current mix while ramping."""
        current = QUANT_MIX["value"]
        report = {}
        for tag, mix in (("", 1.0),) + ((("_current_mix", current),) if current < 1.0 else ()):
            QUANT_MIX["value"] = mix
            report.update({k + tag: v for k, v in held_scores().items()})
        QUANT_MIX["value"] = current
        report["mix"] = current
        report["teacher_ce"] = sum(held["teacher_ce"]) / len(held["teacher_ce"])
        return report

    @torch.no_grad()
    def held_scores():
        model.eval()
        ce, kl = [], []
        for i in range(held["ids"].shape[0]):
            ids = held["ids"][i:i + 1].long().cuda()
            logits = hidden_logits(model, ids)[0, :-1]
            ce.append(float(F.cross_entropy(logits.float(), ids[0, 1:])))
            kl.append(float(topk_kl(logits, held["top_ids"][i].cuda(), held["top_logp"][i].cuda())))
        model.train()
        return {"held_ce": sum(ce) / len(ce), "held_kl": sum(kl) / len(kl)}

    QUANT_MIX["value"] = schedule(step)[0]
    if step == 0:
        report = {"step": 0, **evaluate()}
        print(json.dumps(report), flush=True)
        log.write(json.dumps(report) + "\n")
    generator = torch.Generator().manual_seed(a.seed + step)
    while step < a.steps:
        started = time.perf_counter()
        QUANT_MIX["value"], kl_weight = schedule(step)
        i = int(torch.randint(train["ids"].shape[0], (1,), generator=generator))
        ids = train["ids"][i:i + 1].long().cuda()
        logits = hidden_logits(model, ids)[0, :-1]
        ce = F.cross_entropy(logits.float(), ids[0, 1:])
        kl = topk_kl(logits, train["top_ids"][i].cuda(), train["top_logp"][i].cuda())
        loss = a.ce_weight * ce + kl_weight * kl
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite conversion loss")
        loss.backward()  # the optimizer steps each latent inside backward
        step += 1
        row = {"step": step, "ce": float(ce), "kl": float(kl), "mix": QUANT_MIX["value"], "kl_weight": kl_weight, "seconds": time.perf_counter() - started,
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        if step % a.eval_every == 0 or step == a.steps:
            row.update(evaluate())
        log.write(json.dumps(row) + "\n")
        log.flush()
        if step % 10 == 0 or "held_ce" in row:
            print(json.dumps(row), flush=True)
        # Shared checkpoint policy (train/checkpoint_policy.py, plans/STORAGE_POLICY.md): one rolling resumable
        # slot on a wall-clock cadence or when asked to stop, a weights-only best on the deployed (λ=1) held CE, and
        # at the end the latents alone (the ~23 GB Lion momentum only serves resuming this exact run).
        if "held_ce" in row and policy.save_best({"step": step, "latents": {n: q.detach() for n, q in latents}},
                                                 metric=row["held_ce"]):
            print(json.dumps({"best_weights": step, "held_ce": row["held_ce"]}), flush=True)
        if policy.due() or step == a.steps or (a.checkpoint_every and step % a.checkpoint_every == 0):
            state = {"step": step, "latents": {n: q.detach() for n, q in latents}, "optimizer": optimizer.state_dict()}
            try:
                policy.save(state)
            except OSError as error:
                # DGX-owned disposal (plans/STORAGE_POLICY.md §2): with --drop-slot-when-full the single rolling
                # slot (~47 GB) gives way when the disk cannot hold two; the best weights stay a separate file.
                if error.errno != errno.ENOSPC or not a.drop_slot_when_full:
                    raise
                print(json.dumps({"dropped_rolling_slot_for_space": str(policy.slot), "step": step}), flush=True)
                policy.slot.unlink(missing_ok=True)
                policy.save(state)
            if policy.signaled:
                print(json.dumps({"stopped_on_signal": step}), flush=True)
                return
    policy.finalize({"step": step, "latents": {n: q.detach() for n, q in latents}})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--model", required=True)
    t.add_argument("--text-data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--windows", type=int, default=512)
    t.add_argument("--held", type=int, default=32)
    t.add_argument("--tokens", type=int, default=2048)
    t.add_argument("--k", type=int, default=64)
    t.add_argument("--skip-system", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--extra-glob", action="append", default=[], help="code/prose files packed in (absolute glob)")
    r = sub.add_parser("train")
    r.add_argument("--model", required=True)
    r.add_argument("--teacher", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=2000)
    r.add_argument("--lr", type=float, default=3e-4, help="Lion step in units of each latent's ternary scale")
    r.add_argument("--ce-weight", type=float, default=1.0)
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--ramp-steps", type=int, default=1000, help="ternarization mix ramps 0 → 1 over these updates")
    r.add_argument("--kl-decay-steps", type=int, default=0, help="KL to the BF16 original falls to 0 over these updates")
    r.add_argument("--eval-every", type=int, default=100)
    r.add_argument("--checkpoint-every", type=int, default=0, help="also checkpoint every N updates (0: off)")
    r.add_argument("--checkpoint-minutes", type=float, default=45.0, help="rolling checkpoint cadence (wall clock)")
    r.add_argument("--drop-slot-when-full", action=argparse.BooleanOptionalAction, default=True,
                   help="when the disk cannot hold a second ~47 GB slot, drop the previous one first (DGX)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=True,
                   help="recompute each layer in backward (its ternary values included) instead of keeping them")
    a = p.parse_args(argv)
    (run_teacher if a.phase == "teacher" else run_train)(a)


if __name__ == "__main__":
    main()
