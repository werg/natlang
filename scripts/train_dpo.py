#!/usr/bin/env python3
"""Direct preference optimization of an SFT LoRA adapter on rendered preference pairs
(ts-host/scripts/export-preference-pairs.mjs).

The policy starts as the SFT adapter and keeps training it; the reference is that same adapter, frozen: its log
probabilities are computed once, before training, and stored beside the run. Each step compares a pair's chosen and
rejected completions after their shared prompt:

    loss = -log sigmoid(beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)))
           + nll_weight * (policy's mean negative log likelihood of the chosen completion)

The two sides run one at a time (a long prompt leaves no room for both): the rejected side's log probability first,
without gradients, then each side's backward pass with the loss's exact derivative for it (-beta * sigmoid(-margin)
for the chosen side, +beta * sigmoid(-margin) for the rejected).

Completion characters a pair marks as masked (`chosen_masked` / `rejected_masked`: reasoning no model wrote, or shared by
both sides) are context, not scored. Pairs of programs the SFT run held out (its split.json) are held out here too, and
the preference accuracy and margin on them are reported before and after training.

  docker run --rm --gpus all -v "$PWD:/work" -w /work -e HF_HOME=/work/models/hf natlang-train \\
    python scripts/train_dpo.py runs/spark-lora/v5/preferences.jsonl runs/spark-dpo/v5 \\
      --model models/candidates/spark-x25-4b-train --trust-remote-code --adapter runs/spark-lora/v5-train/checkpoint/weights \\
      --sft-split runs/spark-lora/v5-train/split.json --load-in-4bit --target-modules ... --max-len 8192

Stopping (SIGTERM) finishes the current step and writes a checkpoint; the same command resumes from it.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.corpus import file_digest
from scripts.train_lora import install_stop_handlers, set_layer_checkpointing
from scripts.training_readiness import clip_finite_grad_norm_, require_finite_loss

import torch
import torch.nn.functional as F


def encode(tok, prompt, completion, masked):
    """Prompt (with the masked completion prefix) and the scored completion, as token ids."""
    return (tok(prompt + completion[:masked], add_special_tokens=False)["input_ids"],
            tok(completion[masked:], add_special_tokens=False)["input_ids"])


def sequence_logp(model, side):
    """Summed log probability of a side's completion after its prompt; only the completion's logits are computed."""
    x, y = side
    device = next(model.parameters()).device
    inputs = torch.tensor([x + y], device=device)
    logits = model(input_ids=inputs, logits_to_keep=len(y) + 1).logits[:, :-1].float()
    targets = torch.tensor([y], device=device)
    return torch.gather(F.log_softmax(logits, dim=-1), 2, targets.unsqueeze(-1)).sum()


def dpo_margin(beta, reference, chosen, rejected):
    """beta times how much more the policy than the reference prefers the chosen side."""
    return beta * ((chosen - reference[0]) - (rejected - reference[1]))


def dpo_backward(model, pair, reference, beta, nll_weight=0.0, scale=1.0):
    """Accumulate one pair's loss gradients, times `scale`, a side at a time; its DPO loss and margin."""
    with torch.no_grad():
        rejected = sequence_logp(model, pair["rejected"]).item()
    chosen_logp = sequence_logp(model, pair["chosen"])
    margin = dpo_margin(beta, reference, chosen_logp.item(), rejected)
    weight = beta * torch.sigmoid(torch.tensor(-margin)).item()
    chosen_loss = -weight * chosen_logp - nll_weight * chosen_logp / len(pair["chosen"][1])
    require_finite_loss(chosen_loss)
    (chosen_loss * scale).backward()
    del chosen_logp, chosen_loss
    rejected_loss = weight * sequence_logp(model, pair["rejected"])
    require_finite_loss(rejected_loss)
    (rejected_loss * scale).backward()
    return -F.logsigmoid(torch.tensor(margin)).item(), margin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True, help="the SFT LoRA adapter: the policy's start and the reference")
    ap.add_argument("--sft-split", type=Path, help="the SFT run's split.json: its held-out programs are held out here too")
    ap.add_argument("--holdout", type=int, default=0, help="further pairs held out, whole programs (default: none)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll-weight", type=float, default=0.0,
                    help="weight of the chosen completion's negative log likelihood (keeps it from falling with the rejected)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8, help="pairs per optimizer step")
    ap.add_argument("--max-len", type=int, default=8192, help="pairs whose longer side exceeds this are left out")
    ap.add_argument("--optimizer", choices=("adamw", "paged-adamw-8bit"), default="paged-adamw-8bit")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    stop = install_stop_handlers()
    torch.manual_seed(a.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    rows = [json.loads(line) for line in a.data.open() if line.strip()]
    pairs, too_long = [], 0
    for row in rows:
        chosen = encode(tok, row["prompt"], row["chosen"], row.get("chosen_masked", 0))
        rejected = encode(tok, row["prompt"], row["rejected"], row.get("rejected_masked", 0))
        if max(len(x) + len(y) for x, y in (chosen, rejected)) > a.max_len:
            too_long += 1
            continue
        pairs.append({"id": row["id"], "program_id": row["program_id"], "kind": row.get("kind"),
                      "evidence": row.get("evidence", {}).get("kind"), "chosen": chosen, "rejected": rejected})
    held_programs = set(json.loads(a.sft_split.read_text())["held_programs"]) if a.sft_split else set()
    programs = sorted({p["program_id"] for p in pairs} - held_programs)
    random.Random(a.seed).shuffle(programs)
    extra, count = set(), 0
    for program in programs:
        if count >= a.holdout:
            break
        extra.add(program)
        count += sum(p["program_id"] == program for p in pairs)
    held = [p for p in pairs if p["program_id"] in held_programs | extra]
    train = [p for p in pairs if p["program_id"] not in held_programs | extra]
    steps = math.ceil(len(train) * a.epochs / a.accum)
    print(f"{len(train)} training pairs, {len(held)} held out, {too_long} longer than {a.max_len} tokens; "
          f"{steps} optimizer steps", flush=True)

    load = {"dtype": torch.bfloat16, "trust_remote_code": a.trust_remote_code}
    if a.load_in_4bit:
        load.update({"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)})
    model = AutoModelForCausalLM.from_pretrained(a.model, **load)
    if not a.load_in_4bit:
        model = model.to("cuda")
    model.config.use_cache = False
    ckpt = a.out / "checkpoint"
    state_file = ckpt / "state.json"
    resume = state_file.exists()
    identity = {"data_sha256": file_digest(a.data), "adapter": str(a.adapter), "beta": a.beta, "lr": a.lr,
                "nll_weight": a.nll_weight, "epochs": a.epochs, "accum": a.accum, "max_len": a.max_len, "seed": a.seed,
                "held": sorted(p["id"] for p in held)}
    state = json.loads(state_file.read_text()) if resume else {"step": 0, "log": []}
    if resume and state.get("identity") != identity:
        raise SystemExit("the checkpoint was made with other data or settings; use a new output directory")
    state["identity"] = identity
    a.out.mkdir(parents=True, exist_ok=True)

    # Reference log probabilities: the SFT adapter's, once, before any update.
    reference_file = a.out / "reference.json"
    reference = json.loads(reference_file.read_text()) if reference_file.exists() else {}
    # A resumed run continues its own adapter; its reference was stored when it started.
    model = PeftModel.from_pretrained(model, str(ckpt / "weights" if resume else a.adapter), is_trainable=True)
    if a.gradient_checkpointing:
        set_layer_checkpointing(model, True)
        model.enable_input_require_grads()
    missing = [p for p in pairs if p["id"] not in reference]
    if missing and resume:
        raise SystemExit("reference log probabilities are missing for a resumed run; use a new output directory")
    model.eval()
    with torch.no_grad():
        for n, p in enumerate(missing):
            reference[p["id"]] = [sequence_logp(model, p["chosen"]).item(), sequence_logp(model, p["rejected"]).item()]
            if n % 50 == 0:
                print(f"reference {n}/{len(missing)}", flush=True)
    reference_file.write_text(json.dumps(reference))

    @torch.no_grad()
    def evaluate(label):
        if not held:
            return None
        model.eval()
        margins = [dpo_margin(a.beta, reference[p["id"]], sequence_logp(model, p["chosen"]).item(),
                              sequence_logp(model, p["rejected"]).item()) for p in held]
        result = {"at": label, "accuracy": sum(m > 0 for m in margins) / len(margins),
                  "mean_margin": sum(margins) / len(margins), "pairs": len(margins)}
        print(json.dumps(result), flush=True)
        return result

    if not resume:
        state["heldout_before"] = evaluate("before")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if a.optimizer == "paged-adamw-8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=a.lr)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=a.lr)
    if resume and (ckpt / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt"))
    order = [train[i % len(train)] for i in range(steps * a.accum)]
    random.Random(a.seed).shuffle(order)

    def save():
        model.save_pretrained(ckpt / "weights")
        torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
        state_file.write_text(json.dumps(state, indent=1))

    model.train()
    started = time.time()
    for step in range(state["step"], steps):
        if stop["now"]:
            break
        optimizer.zero_grad(set_to_none=True)
        losses, margins = [], []
        for p in order[step * a.accum:(step + 1) * a.accum]:
            loss, margin = dpo_backward(model, p, reference[p["id"]], a.beta, a.nll_weight, 1 / a.accum)
            losses.append(loss); margins.append(margin)
        clip_finite_grad_norm_(trainable, 1.0)
        # Linear warmup over the first tenth, then constant.
        for group in optimizer.param_groups:
            group["lr"] = a.lr * min(1.0, (step + 1) / max(1, steps // 10))
        optimizer.step()
        state["step"] = step + 1
        state["log"].append({"step": step + 1, "loss": sum(losses) / len(losses),
                             "accuracy": sum(m > 0 for m in margins) / len(margins),
                             "margin": sum(margins) / len(margins), "seconds": round(time.time() - started, 1)})
        print(json.dumps(state["log"][-1]), flush=True)
        if state["step"] % a.save_every == 0:
            save()
    if state["step"] >= steps:
        state["heldout_after"] = evaluate("after")
    save()
    if state["step"] >= steps:
        model.save_pretrained(a.out / "adapter")
        print(f"saved {a.out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
