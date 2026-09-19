#!/usr/bin/env python3
"""Supervised fine-tuning of the interpreter on exported pairs (scripts/export_sft.py), completion-only loss.

Memory-constrained by default so that it runs on a laptop GPU: LoRA adapters on a bf16 base, gradient
checkpointing, one sequence at a time with gradient accumulation, length-capped samples. `--full` trains all
weights instead (needs far more memory).

  docker run --rm --gpus all -v "$PWD:/work" -e HF_HOME=/work/models/hf natlang-train \\
    python scripts/train_lora.py data/sft-v5-small.jsonl runs/lora-v5 --steps 300
"""
import argparse, json, math, random, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=16, help="sequences per optimizer step")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=200, help="pairs kept out for a loss check")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not a.full:
        linear = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n})
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=linear,
                                                 task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    pairs = [json.loads(l) for l in a.data.open()]
    random.shuffle(pairs)

    def encode(p):                       # the pairs are already rendered by the chat template: no special tokens added
        x = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        y = tok(p["completion"], add_special_tokens=False)["input_ids"]
        if len(x) + len(y) > a.max_len:
            return None
        ids = torch.tensor([x + y]).cuda()
        labels = torch.tensor([[-100] * len(x) + y]).cuda()
        return ids, labels

    held, train = pairs[: a.holdout], pairs[a.holdout:]

    @torch.no_grad()
    def heldout_loss():
        model.eval()
        tot = n = 0
        for p in held[:100]:
            e = encode(p)
            if e:
                tot += model(input_ids=e[0], labels=e[1]).loss.item(); n += 1
        model.train()
        return tot / max(n, 1)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    print(f"{len(train)} training pairs; held-out loss before: {heldout_loss():.4f}", flush=True)
    model.train()
    i, skipped, t0 = 0, 0, time.time()
    for step in range(a.steps):
        running = 0.0
        for _ in range(a.accum):
            e = None
            while e is None:
                e = encode(train[i % len(train)]); i += 1
                skipped += e is None
            loss = model(input_ids=e[0], labels=e[1]).loss / a.accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == a.steps - 1:
            print(f"step {step:4d}  loss {running:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  {time.time() - t0:.0f}s", flush=True)
    print(f"held-out loss after: {heldout_loss():.4f}; pairs seen {i}, too long {skipped}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.full:
        model = model.merge_and_unload()
    model.save_pretrained(a.out / "merged", safe_serialization=True)
    tok.save_pretrained(a.out / "merged")
    print(f"saved {a.out / 'merged'}")


if __name__ == "__main__":
    main()
