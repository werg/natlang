"""Distil decision models into the student's decision readout (the runtime's `readout: decision`).

Rows come from `ts-host/scripts/skills/export-decision-prompts.mjs`: the exact messages and options a typed decision
function sends, the gold distribution, and teacher distributions (Strands Decider, Clef). Training scores every
option with the reference server's `decision_logprobs` (one prompt pass, options continued from its cache), so the
readout trained here is the one the runtime calls, and minimises the cross-entropy of the normalised option
distribution against the chosen target (backpropagated one option at a time):

  --target gold              the dataset label (one-hot, or the soft/fractional label where the source has one)
  --target teacher:NAME      a decision model's full distribution (distillation)
  --target mix:NAME:W        W * teacher + (1 - W) * gold

`--steps 0` evaluates the untrained readout. Evaluation is on held-out rows by family: Brier (choice and yes/no),
ranked probability score (ordered levels), log loss and top-1 agreement with the gold argmax, plus the teacher's own
scores on the same rows when it labelled them. Only LoRA parameters train; the CUDA allocation is capped.

  python -m natlang_neuralese.train.decision --prompts prompts.jsonl --out run/ --target teacher:strands-decider-2B-hobson-v19
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path


def target_distribution(row: dict, target: str) -> list[float] | None:
    gold = [float(v) for v in row["gold"]]
    if target == "gold":
        return gold
    kind, _, rest = target.partition(":")
    if kind == "teacher":
        return row["teachers"].get(rest)
    if kind == "mix":
        name, _, weight = rest.rpartition(":")
        teacher = row["teachers"].get(name)
        if teacher is None:
            return None
        w = float(weight)
        return [w * t + (1 - w) * g for t, g in zip(teacher, gold)]
    raise ValueError(f"unknown target {target!r}")


def scores(kind: str, predicted: list[float], gold: list[float]) -> dict:
    eps = 1e-12
    out = {"log_loss": -sum(g * math.log(max(p, eps)) for p, g in zip(predicted, gold)),
           "top1": float(max(range(len(predicted)), key=predicted.__getitem__) == max(range(len(gold)), key=gold.__getitem__))}
    if kind == "score":
        cp = cg = 0.0
        total = 0.0
        for p, g in list(zip(predicted, gold))[:-1]:
            cp, cg = cp + p, cg + g
            total += (cp - cg) ** 2
        out["rps"] = total / max(1, len(gold) - 1)
    else:
        out["brier"] = sum((p - g) ** 2 for p, g in zip(predicted, gold))
    return out


def summarize(rows: list[dict]) -> dict:
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row["scores"].items():
            families[row["family"]][key].append(value)
        for key, value in (row.get("teacher_scores") or {}).items():
            families[row["family"]]["teacher_" + key].append(value)
    return {family: {key: round(sum(v) / len(v), 4) for key, v in metrics.items()} | {"n": len(next(iter(metrics.values())))}
            for family, metrics in sorted(families.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", default="gold")
    parser.add_argument("--teacher-eval", default=None, help="teacher whose own held-out scores are reported")
    parser.add_argument("--base", default=None, help="base model (default: the port's LFM2.5-350M)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8, help="cases per optimiser step (gradient accumulation)")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="save a resumable checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="continue from OUT/checkpoint.pt (a stopped run)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--eval-per-family", type=int, default=40)
    parser.add_argument("--max-options", type=int, default=0, help="skip cases with more options (0: no limit)")
    parser.add_argument("--train-options", type=int, default=16,
                        help="options per training case: those with target mass first, then random others "
                             "(a sampled softmax; evaluation always uses every option; 0: all)")
    parser.add_argument("--require-teacher", default=None,
                        help="train only on rows this teacher labelled, so gold and teacher arms see the same cases")
    parser.add_argument("--soft-prompts", default=None,
                        help="system-prompt bank (.nz): its pieces in the prompts become trainable (DECISIONS.md 40)")
    parser.add_argument("--prompt-lr", type=float, default=1e-3, help="learning rate of the soft prompt pieces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=float(os.environ.get("NATLANG_CUDA_MEMORY_GB", 16)))
    args = parser.parse_args(argv)

    import torch

    from ..serve import load_engine
    from ..serve.grad import GradSession
    from ..prompt_bank import soften
    from .adapters import inject_lora, lora_state

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, held = [], defaultdict(list)
    with open(args.prompts) as stream:
        for line in stream:
            row = json.loads(line)
            if args.max_options and len(row["options"]) > args.max_options:
                continue
            if row["role"] == "heldout":
                held[row["family"]].append(row)
            elif target_distribution(row, args.target) is not None and (
                    not args.require_teacher or args.require_teacher in row["teachers"]):
                train.append(row)
    evaluation = [row for rows in held.values() for row in rows[:args.eval_per_family]]
    random.shuffle(train)
    engine = load_engine(args.base, device=args.device)
    backbone = engine.backbone
    parameters = []
    if args.steps and args.rank:
        groups = inject_lora(backbone, list(range(backbone.num_layers)), rank=args.rank, alpha=2 * args.rank)
        parameters = [p for ps in groups.values() for p in ps]
    # Soft prompt pieces: gradient leaves under the bank's block IDs, trained with their own learning rate.
    bank, prompts, leaves, used = None, {}, {}, set()
    if args.soft_prompts:
        from ..prompt_bank import load_bank
        bank = load_bank(args.soft_prompts)
        if bank.dialect != engine.dialect:
            raise SystemExit(f"the bank is in {bank.dialect}, the model speaks {engine.dialect}")
        prompts = {piece: torch.nn.Parameter(rows.to(engine.device)) for piece, rows in bank.rows.items()}
        leaves = {bank.ids[piece]: param for piece, param in prompts.items()}
    session = GradSession(engine)
    groups_ = ([{"params": parameters, "lr": args.lr}] if parameters else []) + (
        [{"params": list(prompts.values()), "lr": args.prompt_lr}] if prompts and args.steps else [])
    optimizer = torch.optim.AdamW(groups_, weight_decay=0.0) if groups_ else None
    trainable = parameters + (list(prompts.values()) if args.steps else [])

    def messages_of(row):
        if not bank:
            return row["messages"]
        softened, pieces = soften(row["messages"], bank.texts, bank.ids)
        used.update(pieces)
        return softened
    config = {k: v for k, v in vars(args).items()} | {"train_rows": len(train), "eval_rows": len(evaluation)}
    started_step, cursor = 0, 0
    checkpoint_path = out / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        # Same seed, same shuffled order: the cursor, RNG state, LoRA weights and optimiser state continue the run.
        saved = torch.load(checkpoint_path, map_location=engine.device, weights_only=False)
        named = dict(backbone.hf.named_parameters())
        with torch.no_grad():
            for name, value in saved["lora"].items():
                named[name].copy_(value.to(named[name].device))
            for piece, value in saved.get("prompts", {}).items():
                prompts[piece].copy_(value.to(engine.device))
        optimizer.load_state_dict(saved["optimizer"])
        started_step, cursor = saved["step"] + 1, saved["cursor"]
        random.setstate(saved["random"])
        print(json.dumps({"resumed": started_step}), flush=True)
    else:
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = open(out / "train.jsonl", "a" if started_step else "w")

    def readout(row):
        logp, _ = session.decision_logprobs(messages_of(row), None, row["options"], leaves)
        return torch.log_softmax(logp, 0)

    started = time.time()
    backbone.train(False)
    for step in range(started_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.batch):
            if not train:
                break
            row = train[cursor % len(train)]
            cursor += 1
            goal_list, options = target_distribution(row, args.target), row["options"]
            if args.train_options and len(options) > args.train_options:
                ranked = sorted(range(len(options)), key=lambda i: -goal_list[i])
                massive = [i for i in ranked if goal_list[i] > 0][:args.train_options]
                rest = [i for i in range(len(options)) if i not in massive]
                keep = sorted(massive + random.sample(rest, args.train_options - len(massive)))
                goal_list, options = [goal_list[i] for i in keep], [options[i] for i in keep]
            goal = torch.tensor(goal_list, device=engine.device)
            # One option's graph at a time, so many-option families fit (GradSession.decision_backward).
            with torch.enable_grad():
                losses.append(session.decision_backward(messages_of(row), None, options, goal, 1 / args.batch, leaves))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 10 == 0 or step == args.steps - 1:
            record = {"step": step, "loss": sum(losses) / max(1, len(losses)), "seconds": round(time.time() - started),
                      "epoch": round(cursor / max(1, len(train)), 3)}
            if args.device.startswith("cuda"):
                record["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0 and step + 1 < args.steps:
            torch.save({"lora": lora_state(backbone), "optimizer": optimizer.state_dict(), "step": step, "cursor": cursor,
                        "random": random.getstate(), "prompts": {k: v.detach() for k, v in prompts.items()}},
                       checkpoint_path.with_suffix(".tmp"))
            checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
    if parameters:
        torch.save({"lora": lora_state(backbone), "rank": args.rank, "config": config}, out / "adapter.pt")
    if bank and args.steps:
        from ..prompt_bank import save_bank
        save_bank(out / "system-prompts.nz", bank, {piece: prompts[piece] for piece in used},
                            {"kind": "system-prompt-bank", "init": args.soft_prompts, "trained_by": "natlang_neuralese.train.decision",
                             "pieces_trained": sorted(used), "steps": args.steps, "prompt_lr": args.prompt_lr})
        print(json.dumps({"system_prompts": str(out / "system-prompts.nz"), "trained": sorted(used)}), flush=True)

    rows = []
    with torch.no_grad():
        for row in evaluation:
            predicted = readout(row).exp().tolist()
            result = {"id": row["id"], "family": row["family"], "kind": row["kind"], "predicted": predicted,
                      "scores": scores(row["kind"], predicted, row["gold"])}
            teacher = row["teachers"].get(args.teacher_eval) if args.teacher_eval else None
            if teacher:
                result["teacher_scores"] = scores(row["kind"], teacher, row["gold"])
            rows.append(result)
    with open(out / "eval.jsonl", "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    summary = {"target": args.target, "steps": args.steps, "families": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary)[:2000], flush=True)


if __name__ == "__main__":
    main()
