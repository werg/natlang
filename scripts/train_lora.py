#!/usr/bin/env python3
"""Supervised fine-tuning of the interpreter on exported pairs (scripts/export_sft.py), completion-only loss.

Memory-constrained by default so that it runs on a laptop GPU: LoRA adapters on a bf16 base, gradient
checkpointing, one sequence at a time with gradient accumulation, length-capped samples. `--full` trains all
weights instead (needs far more memory).

Every run is stoppable and resumable. A checkpoint (adapter or full weights, optimizer, scheduler, step, position
in the data, random state) is written every `--save-every` optimizer steps and when the process is asked to stop
(SIGTERM or Ctrl-C: it finishes the current step first). Starting the same command again continues from the
checkpoint; the data order is a function of `--seed`, so a resumed run sees exactly the pairs it would have seen.
A servable model can be exported from the latest checkpoint at any time, also while no training is running:

  docker run --rm --gpus all -v "$PWD:/work" -e HF_HOME=/work/models/hf --name natlang-train natlang-train \\
    python scripts/train_lora.py data/sft.jsonl runs/lora --steps 600          # start, or resume
  docker stop -t 120 natlang-train                                             # stop cleanly (a checkpoint is written)
  ... python scripts/train_lora.py data/sft.jsonl runs/lora --merge-only       # runs/lora/merged from the checkpoint
"""
import argparse, json, math, os, random, shutil, signal, time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps in total (a resumed run continues up to this)")
    ap.add_argument("--accum", type=int, default=16, help="sequences per optimizer step")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=200, help="pairs kept out for a loss check")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint every N optimizer steps")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--merge-only", action="store_true", help="export out/merged from the latest checkpoint and exit")
    a = ap.parse_args()
    ckpt = a.out / "checkpoint"
    state_file = ckpt / "state.json"
    if a.fresh and ckpt.exists():
        shutil.rmtree(ckpt)
    resume = state_file.exists()
    state = json.loads(state_file.read_text()) if resume else {"step": 0, "cursor": 0, "skipped": 0, "log": []}
    if a.merge_only and not resume:
        raise SystemExit(f"no checkpoint in {ckpt}")

    tok = AutoTokenizer.from_pretrained(a.model)
    base_src = str(ckpt / "weights") if (resume and a.full) else a.model
    model = AutoModelForCausalLM.from_pretrained(base_src, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not a.full:
        if resume:
            model = PeftModel.from_pretrained(model, str(ckpt / "weights"), is_trainable=True)
        else:
            linear = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n})
            model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=linear,
                                                     task_type="CAUSAL_LM"))
        model.enable_input_require_grads()

    def export_merged():
        merged = model.merge_and_unload() if not a.full else model
        merged.save_pretrained(a.out / "merged", safe_serialization=True)
        tok.save_pretrained(a.out / "merged")
        (a.out / "merged" / "natlang_training.json").write_text(json.dumps({k: state[k] for k in ("step", "cursor")} | {"data": str(a.data)}))
        print(f"saved {a.out / 'merged'} (step {state['step']})", flush=True)

    if a.merge_only:
        export_merged()
        return

    pairs = [json.loads(l) for l in a.data.open()]
    random.Random(a.seed).shuffle(pairs)                   # the order depends on the seed only: resumable
    held, train = pairs[: a.holdout], pairs[a.holdout:]

    def encode(p):                       # the pairs are already rendered by the chat template: no special tokens added
        x = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        y = tok(p["completion"], add_special_tokens=False)["input_ids"]
        if len(x) + len(y) > a.max_len:
            return None
        return torch.tensor([x + y]).cuda(), torch.tensor([[-100] * len(x) + y]).cuda()

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
    if resume:
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cuda"))
        sched.load_state_dict(torch.load(ckpt / "scheduler.pt"))
        torch.set_rng_state(torch.load(ckpt / "rng.pt"))
        print(f"resumed from step {state['step']} (pair {state['cursor']})", flush=True)
    else:
        torch.manual_seed(a.seed)
        state["heldout_before"] = heldout_loss()
        print(f"{len(train)} training pairs; held-out loss before: {state['heldout_before']:.4f}", flush=True)

    def save_checkpoint():
        tmp = a.out / "checkpoint.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        model.save_pretrained(tmp / "weights", safe_serialization=True)
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        torch.save(sched.state_dict(), tmp / "scheduler.pt")
        torch.save(torch.get_rng_state(), tmp / "rng.pt")
        (tmp / "state.json").write_text(json.dumps({**state, "args": {k: str(v) for k, v in vars(a).items()}}))
        if ckpt.exists():                                  # swap in atomically enough: the old one goes last
            old = a.out / "checkpoint.old"
            if old.exists():
                shutil.rmtree(old)
            os.rename(ckpt, old)
            os.rename(tmp, ckpt)
            shutil.rmtree(old)
        else:
            os.rename(tmp, ckpt)

    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(now=True))
    a.out.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.time()
    while state["step"] < a.steps and not stop["now"]:
        running = 0.0
        for _ in range(a.accum):
            e = None
            while e is None:
                e = encode(train[state["cursor"] % len(train)]); state["cursor"] += 1
                state["skipped"] += e is None
            loss = model(input_ids=e[0], labels=e[1]).loss / a.accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        state["step"] += 1
        if state["step"] % 10 == 0 or state["step"] == a.steps:
            state["log"].append([state["step"], round(running, 4)])
            print(f"step {state['step']:4d}  loss {running:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  {time.time() - t0:.0f}s", flush=True)
        if state["step"] % a.save_every == 0:
            save_checkpoint()
    save_checkpoint()
    if stop["now"]:
        print(f"stopped at step {state['step']} of {a.steps}; checkpoint written. Run the same command to continue, "
              f"or --merge-only to export this state.", flush=True)
        return
    state["heldout_after"] = heldout_loss()
    print(f"held-out loss after: {state['heldout_after']:.4f}; pairs seen {state['cursor']}, too long {state['skipped']}", flush=True)
    save_checkpoint()
    export_merged()


if __name__ == "__main__":
    main()
