"""Full-latent QAT conversion of a BF16 Maple-family model (Mellum) to ternary (plans/mellum-port.md, owner 2026-10-09).

Every attention projection and every expert matrix becomes a BF16 latent whose forward value is its ternary codes
(Maple's per-row rule: threshold 0.7 x row absmean, scale = mean kept magnitude; straight-through gradient). Router,
norms, embedding and head stay floating, as in Maple's deployed form. Recovery: CE on our corpus plus a sparse KL to
the BF16 original's top-k next-token distribution, dumped once beforehand (``teacher`` phase), so the original never
has to sit next to the student. The KL to the original belongs to this conversion only (owner: no anchor may cap
learning afterwards).

    python -m natlang_neuralese.maple.qat_convert teacher --model DIR --text-data T --out DIR [--windows 512]
    python -m natlang_neuralese.maple.qat_convert train --model DIR --teacher DIR --out DIR [--steps 2000]

Conversion v3 (records): the teacher phase takes token-id records of the BF16 model's own responses in its own chat
template (`teacher --records RECORDS.jsonl`, maple/distill_data.py) and the trainer distills KL only, prompt positions
down-weighted, with a generation gate (`train --probe PROBE.jsonl --ce-weight 0`).
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


def _record_split(records: list[dict], held: int) -> tuple[list[dict], list[dict]]:
    """A deterministic held-out set: the ``held`` records with the smallest id hashes."""
    import hashlib

    ranked = sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    held_ids = {r["id"] for r in ranked[:held]}
    return [r for r in records if r["id"] not in held_ids], [r for r in records if r["id"] in held_ids]


@torch.no_grad()
def _teacher_topk(model, ids: torch.Tensor, k: int, chunk: int = 2048):
    """Top-k next-token log-probabilities of every position but the last, computed in position chunks."""
    h = model.model(input_ids=ids).last_hidden_state[0, :-1]
    top_ids, top_logp, token_logp = [], [], []
    for start in range(0, h.shape[0], chunk):
        logp = torch.log_softmax(model.lm_head(h[start:start + chunk]).float(), -1)
        values, index = logp.topk(k, -1)
        top_ids.append(index.int().cpu())
        top_logp.append(values.half().cpu())
        token_logp.append(logp.gather(-1, ids[0, 1 + start:1 + start + chunk, None])[:, 0].cpu())
    return torch.cat(top_ids), torch.cat(top_logp), torch.cat(token_logp)


def run_teacher_records(a):
    """Teacher phase over token-id records (maple/distill_data.py: the BF16 model's own responses in its own chat
    template): top-k on the whole sequence, prompt and response, in shards of ``--shard`` records (resumable)."""
    model = load_maple(a.model, device="cuda", dtype=torch.bfloat16, ternary_attention=False).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = [r for r in map(json.loads, open(a.records))
               if r["response_ids"] and len(r["prompt_ids"]) + len(r["response_ids"]) <= a.tokens]
    train, held = _record_split(records, a.held)
    shards = {}
    for split, rows in (("test", held), ("train", train)):
        shards[split] = []
        for number, start in enumerate(range(0, len(rows), a.shard)):
            name = f"{split}-{number:04d}.pt"
            shards[split].append(name)
            if (out / name).exists():
                continue
            shard = {"id": [], "source": [], "ids": [], "prompt_len": [], "top_ids": [], "top_logp": [], "teacher_ce": [],
                     "enable_thinking": []}
            for row in rows[start:start + a.shard]:
                ids = torch.tensor([row["prompt_ids"] + row["response_ids"]], device="cuda")
                top_ids, top_logp, token_logp = _teacher_topk(model, ids, a.k)
                response = token_logp[len(row["prompt_ids"]) - 1:]
                shard["id"].append(row["id"])
                shard["source"].append(row["source"])
                shard["enable_thinking"].append(row.get("enable_thinking", False))
                shard["ids"].append(ids[0].int().cpu())
                shard["prompt_len"].append(len(row["prompt_ids"]))
                shard["top_ids"].append(top_ids)
                shard["top_logp"].append(top_logp)
                shard["teacher_ce"].append(float(-response.mean()))
            torch.save(shard, out / (name + ".pending"))
            (out / (name + ".pending")).replace(out / name)
            print(json.dumps({"shard": name, "records": len(shard["ids"]),
                              "positions": sum(int(t.numel()) for t in shard["ids"])}), flush=True)
    (out / "teacher.json").write_text(json.dumps({"format": "records", "model": a.model, "records": a.records,
                                                  "k": a.k, "tokens": a.tokens, "train": len(train), "test": len(held),
                                                  "shards": shards}, indent=1) + "\n")


def _load_record_shards(teacher_dir: Path, names: list[str]) -> dict:
    merged: dict = {}
    for name in names:
        shard = torch.load(teacher_dir / name, map_location="cpu", mmap=True)
        for key, value in shard.items():
            merged.setdefault(key, []).extend(value)
    return merged


def weighted_topk_kl(model, ids: torch.Tensor, top_ids: torch.Tensor, top_logp: torch.Tensor, prompt_len: int,
                     prompt_weight: float, chunk: int = 1024):
    """KL(teacher || student) over the teacher's top-k at every position (prompt positions weighted ``prompt_weight``,
    response positions 1), plus the student's CE on the response tokens. The head and softmax run per position chunk
    under checkpointing, so a long sequence never holds full-vocabulary logits for every position at once."""
    from torch.utils.checkpoint import checkpoint

    h = model.model(input_ids=ids).last_hidden_state[0, :-1]
    targets = ids[0, 1:]
    positions = h.shape[0]
    weight = torch.ones(positions, device=h.device)
    weight[:max(0, prompt_len - 1)] = prompt_weight

    def piece(hidden, t_ids, t_logp, target):
        logp = torch.log_softmax(model.lm_head(hidden).float(), -1)
        t = t_logp.float()
        t = t - torch.logsumexp(t, -1, keepdim=True)
        kl = (t.exp() * (t - logp.gather(-1, t_ids.long()))).sum(-1)
        return kl, -logp.gather(-1, target[:, None])[:, 0], (logp.argmax(-1) == t_ids[:, 0].long()).float()

    kls, ces, agree = [], [], []
    for start in range(0, positions, chunk):
        end = start + chunk
        kl, ce, match = checkpoint(piece, h[start:end], top_ids[start:end].to(h.device), top_logp[start:end].to(h.device),
                                   targets[start:end], use_reentrant=False)
        kls.append(kl), ces.append(ce), agree.append(match)
    kl, ce, agree = torch.cat(kls), torch.cat(ces), torch.cat(agree)
    response = slice(max(0, prompt_len - 1), positions)
    return {"loss_kl": (weight * kl).sum() / weight.sum(), "response_kl": kl[response].mean(),
            "prompt_kl": kl[:max(1, prompt_len - 1)].mean(), "response_ce": ce[response].mean(),
            "response_agreement": agree[response].mean()}


def _save_slot(a, policy, state, metric=None):
    """The full state (a declared checkpoint point, a stop, the end); ``metric``: the evaluation of that point, which
    publishes this very file as the best when it is the best so far (no separate best write)."""
    try:
        policy.save(state, metric=metric)
    except OSError as error:
        # DGX-owned disposal (plans/STORAGE_POLICY.md §2): with --drop-slot-when-full the single rolling slot (~47 GB)
        # gives way when the disk cannot hold two; a best linked to it keeps that inode (and its space).
        if error.errno != errno.ENOSPC or not a.drop_slot_when_full:
            raise
        print(json.dumps({"dropped_rolling_slot_for_space": str(policy.slot), "step": state["step"]}), flush=True)
        policy.slot.unlink(missing_ok=True)
        policy.save(state, metric=metric)


def run_train_records(a, teacher_dir: Path, meta: dict):
    """Conversion v3: KL to the BF16 model only, on its own responses in its own chat template. Prompt positions are
    weighted ``--prompt-weight``, response positions 1; records are drawn in shuffled epochs. At every ``--eval-every``
    point: held KL/agreement of the deployed model (λ=1, and the current mix while ramping) and the generation gate
    (maple/generation_gate.py) on fixed probes. After the ramp, two consecutive gates more than ``--gate-drop`` below
    the best post-ramp gate stop the run (its last state is checkpointed)."""
    from transformers import AutoTokenizer

    from .distill_data import render
    from .generation_gate import run_gate

    train = _load_record_shards(teacher_dir, meta["shards"]["train"])
    held = _load_record_shards(teacher_dir, meta["shards"]["test"])
    held_rows = list(range(min(a.held_eval, len(held["ids"]))))
    probes = [json.loads(line) for line in open(a.probe)] if a.probe else []
    tokenizer = AutoTokenizer.from_pretrained(a.model) if probes else None
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>") if probes else None
    model, latents, optimizer, policy, step, out, log = _setup_training(a)
    n = len(train["ids"])

    def order(epoch):
        return torch.randperm(n, generator=torch.Generator().manual_seed(a.seed + epoch)).tolist()

    def schedule(at):
        return (min(1.0, at / a.ramp_steps) if a.ramp_steps else 1.0), a.kl_weight

    @torch.no_grad()
    def held_scores():
        model.eval()
        sums: dict[str, float] = {}
        for i in held_rows:
            ids = held["ids"][i][None].long().cuda()
            scores = weighted_topk_kl(model, ids, held["top_ids"][i], held["top_logp"][i], held["prompt_len"][i],
                                      a.prompt_weight)
            for key, value in scores.items():
                sums[key] = sums.get(key, 0.0) + float(value)
        model.train()
        return {"held_" + k.replace("loss_", ""): v / len(held_rows) for k, v in sums.items()}

    def evaluate(gate: bool, mixes):
        current = QUANT_MIX["value"]
        report = {"mix": current}
        for tag, mix in mixes:
            QUANT_MIX["value"] = mix
            report.update({k + tag: v for k, v in held_scores().items()})
            if gate and probes and (mix == 1.0 or tag == "_bf16"):  # the gate judges the deployed model
                model.eval()
                result = run_gate(model, tokenizer, probes, render, eos, keep_text=a.gate_text)
                model.train()
                report.update({"gate_pass" + tag: result["gate_pass"], "gate_checks" + tag: result["gate_checks"],
                               "gate_kinds" + tag: result["gate_kinds"]})
                with open(out / "gate.jsonl", "a") as handle:
                    handle.write(json.dumps({"step": step, "mix": mix, **result}) + "\n")
        QUANT_MIX["value"] = current
        report["teacher_response_ce"] = sum(held["teacher_ce"][i] for i in held_rows) / len(held_rows)
        return report

    QUANT_MIX["value"] = schedule(step)[0]
    if step == 0:  # mix 0 is the BF16 model exactly: the gate's reference; mix 1 is naive ternarization
        report = {"step": 0, **evaluate(True, (("_bf16", 0.0), ("", 1.0)))}
        print(json.dumps({k: v for k, v in report.items() if "checks" not in k}), flush=True)
        log.write(json.dumps(report) + "\n")
        log.flush()
    from ..train.loop import Cadence
    evals = Cadence(a.eval_every)  # declared step points only (plans/STORAGE_POLICY.md)
    best_gate, low_gates = None, 0
    while step < a.steps:
        started = time.perf_counter()
        QUANT_MIX["value"], kl_weight = schedule(step)
        i = order(step // n)[step % n]
        ids = train["ids"][i][None].long().cuda()
        scores = weighted_topk_kl(model, ids, train["top_ids"][i], train["top_logp"][i], train["prompt_len"][i],
                                  a.prompt_weight)
        loss = kl_weight * scores["loss_kl"] + a.ce_weight * scores["response_ce"]
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite conversion loss")
        loss.backward()  # the optimizer steps each latent inside backward
        step += 1
        scores = {k: float(v.detach()) for k, v in scores.items()}
        row = {"step": step, "kl": scores["loss_kl"], "response_kl": scores["response_kl"],
               "response_ce": scores["response_ce"], "agreement": scores["response_agreement"],
               "tokens": int(ids.shape[1]), "record": train["id"][i], "mix": QUANT_MIX["value"],
               "seconds": time.perf_counter() - started, "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        stop = None
        write = policy.due(step, end=step == a.steps)
        if not policy.signaled and (evals.due(step) or step == a.steps):  # the end: the gate
            mixes = (("", 1.0),) + ((("_current_mix", QUANT_MIX["value"]),) if QUANT_MIX["value"] < 1.0 else ())
            row.update(evaluate(True, mixes))
            if "gate_pass" in row and step >= a.ramp_steps:
                best_gate = row["gate_pass"] if best_gate is None else max(best_gate, row["gate_pass"])
                low_gates = low_gates + 1 if row["gate_pass"] < best_gate - a.gate_drop else 0
                if low_gates >= 2:
                    stop = "gate degraded twice after the ramp"
        log.write(json.dumps(row) + "\n")
        log.flush()
        if step % 10 == 0 or "held_kl" in row:
            print(json.dumps({k: v for k, v in row.items() if "checks" not in k}), flush=True)
        # Best deployed weights: gate first, then held KL; judged only at full-state writes.
        metric = (1.0 - row.get("gate_pass", 0.0)) + row["held_kl"] if "held_kl" in row else None
        if stop or write:
            _save_slot(a, policy, {"step": step, "latents": {k: q.detach() for k, q in latents},
                                   "optimizer": optimizer.state_dict()}, None if policy.signaled else metric)
            if policy.signaled or stop:
                print(json.dumps({"stopped": stop or "signal", "step": step}), flush=True)
                return


def _setup_training(a):
    """The student (full-latent QAT on every attention and expert matrix), Lion stepping inside backward, the shared
    checkpoint policy and the resume point. Returns (model, latents, optimizer, policy, step, out, log)."""
    from ..train.optim import LionSR

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
    policy = CheckpointPolicy(out, every_steps=a.checkpoint_every).install_signal_handlers()
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
    return model, latents, optimizer, policy, step, out, log


def run_train(a):
    teacher_dir = Path(a.teacher)
    meta = json.loads((teacher_dir / "teacher.json").read_text())
    if meta.get("format") == "records":
        return run_train_records(a, teacher_dir, meta)
    train, held = torch.load(teacher_dir / "train.pt"), torch.load(teacher_dir / "test.pt")
    model, latents, optimizer, policy, step, out, log = _setup_training(a)

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
    from ..train.loop import Cadence
    evals = Cadence(a.eval_every)  # declared step points only (plans/STORAGE_POLICY.md)
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
        write = policy.due(step, end=step == a.steps)
        if not policy.signaled and (evals.due(step) or step == a.steps):  # the end: the gate
            row.update(evaluate())
        log.write(json.dumps(row) + "\n")
        log.flush()
        if step % 10 == 0 or "held_ce" in row:
            print(json.dumps(row), flush=True)
        # Shared checkpoint policy (train/checkpoint_policy.py, plans/STORAGE_POLICY.md): full state at the declared
        # checkpoint points, on a stop and at the end; the best (deployed λ=1 held CE) is one of those writes.
        if write:
            _save_slot(a, policy, {"step": step, "latents": {n: q.detach() for n, q in latents},
                                   "optimizer": optimizer.state_dict()},
                       None if policy.signaled else row.get("held_ce"))
            if policy.signaled:
                print(json.dumps({"stopped_on_signal": step}), flush=True)
                return


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--model", required=True)
    t.add_argument("--text-data")
    t.add_argument("--out", required=True)
    t.add_argument("--windows", type=int, default=512)
    t.add_argument("--held", type=int, default=32)
    t.add_argument("--tokens", type=int, default=2048)
    t.add_argument("--k", type=int, default=64)
    t.add_argument("--skip-system", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--extra-glob", action="append", default=[], help="code/prose files packed in (absolute glob)")
    t.add_argument("--records", help="token-id records (maple/distill_data.py generate); replaces --text-data windows")
    t.add_argument("--shard", type=int, default=256, help="records per teacher shard (--records)")
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
    r.add_argument("--eval-every", type=int, default=100,
                   help="evaluation points: every N updates, and the end (the gate; recipes make --steps a multiple)")
    r.add_argument("--checkpoint-every", type=int, default=0,
                   help="full-state points: every N updates, a multiple of --eval-every (0: end and stops only)")
    r.add_argument("--drop-slot-when-full", action=argparse.BooleanOptionalAction, default=True,
                   help="when the disk cannot hold a second ~47 GB slot, drop the previous one first (DGX)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--prompt-weight", type=float, default=0.25, help="KL weight of prompt positions (records teacher)")
    r.add_argument("--probe", help="generation-gate probes (maple/distill_data.py prompts --probe)")
    r.add_argument("--held-eval", type=int, default=48, help="held records scored at each evaluation (records teacher)")
    r.add_argument("--gate-drop", type=float, default=0.15, help="stop after two post-ramp gates this far below the best")
    r.add_argument("--gate-text", action="store_true", help="keep generated text in gate.jsonl")
    r.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=True,
                   help="recompute each layer in backward (its ternary values included) instead of keeping them")
    a = p.parse_args(argv)
    if a.phase == "teacher":
        if bool(a.records) == bool(a.text_data):
            p.error("teacher needs exactly one of --records and --text-data")
        (run_teacher_records if a.records else run_teacher)(a)
    else:
        run_train(a)


if __name__ == "__main__":
    main()
