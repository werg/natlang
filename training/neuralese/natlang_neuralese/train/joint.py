"""Joint training of a teacher and a student that share a tokenizer, with exact distillation (MAPLE_QWEN_JOINT §2).

One process, one token batch per micro-step: the teacher (QAT LoRA on its attention projections, or frozen) takes its
own cross-entropy; the student (full fine-tune) takes cross-entropy plus a ramped KL to the teacher's detached
distribution. Both losses go through ``chunked_ce_kl``. Input rows are rendered SFT turns (``prompt`` +
``completion`` strings in the shared chat format); only completion tokens are supervised.

    python -m natlang_neuralese.train.joint --teacher Qwen/Qwen3-1.7B --student /home/werg/data/models/qwen3-0.6b \\
        --data runs/maple-joint-20261005/qwen3-render-v1.jsonl --out runs/maple-joint-20261005/smoke-v1 --steps 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from ..maple.ternary import add_qat_lora, qat_adapters
from .joint_kd import IGNORE, chunked_ce_kl, kl_ramp, require_shared_tokenizer

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


def model_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))


def load_rows(path, tokenizer, max_length, split):
    rows = []
    for line in open(path):
        row = json.loads(line)
        if split and row.get("split", "train") != split:
            continue
        prompt = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        ids = (prompt + completion)[-max_length:]
        supervised = min(len(completion), len(ids))
        labels = [IGNORE] * (len(ids) - supervised) + ids[len(ids) - supervised:]
        rows.append((ids, labels))
    return rows


def final_hidden(model, ids):
    return model.model(input_ids=ids).last_hidden_state


def output_weight(model):
    return model.get_output_embeddings().weight


def shifted(ids, labels, device):
    ids_t = torch.tensor([ids], device=device)
    target = torch.tensor([labels[1:] + [IGNORE]], device=device)
    return ids_t, target


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--teacher-mode", choices=["qat", "lora", "frozen"], default="qat",
                    help="qat: ternary QAT LoRA (Maple); lora: plain LoRA (stand-in teachers); frozen")
    ap.add_argument("--teacher-rank", type=int, default=8)
    ap.add_argument("--teacher-lr", type=float, default=1e-4)
    ap.add_argument("--student-lr", type=float, default=1e-5)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-ramp", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--eval-rows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    teacher_dir, student_dir = model_dir(args.teacher), model_dir(args.student)
    fingerprint = require_shared_tokenizer(teacher_dir, student_dir)
    tokenizer = AutoTokenizer.from_pretrained(student_dir)
    device = "cuda"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.data, tokenizer, args.max_length, "train")
    random.shuffle(rows)
    held, train = rows[:args.eval_rows], rows[args.eval_rows:]

    teacher = AutoModelForCausalLM.from_pretrained(teacher_dir, dtype=torch.bfloat16).to(device)
    teacher.requires_grad_(False)
    teacher_params = []
    if args.teacher_mode != "frozen":
        for name, module in teacher.named_modules():
            if name.rsplit(".", 1)[-1] in ATTENTION:
                add_qat_lora(module, rank=args.teacher_rank, alpha=2 * args.teacher_rank,
                             quantize=args.teacher_mode == "qat")
        teacher_params = [p for a in qat_adapters(teacher).values() for p in a.parameters()]
    student = AutoModelForCausalLM.from_pretrained(student_dir, dtype=torch.float32).to(device)
    student.train()
    teacher.train(args.teacher_mode != "frozen")

    student_opt = torch.optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=0.0)
    teacher_opt = torch.optim.AdamW(teacher_params, lr=args.teacher_lr) if teacher_params else None

    def evaluate():
        totals = {"teacher_ce": 0.0, "student_ce": 0.0, "kl": 0.0, "tokens": 0}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ids, labels in held:
                x, y = shifted(ids, labels, device)
                ht = final_hidden(teacher, x)
                _, tp = chunked_ce_kl(ht, output_weight(teacher), y, chunk=args.chunk)
                hs = final_hidden(student, x)
                _, sp = chunked_ce_kl(hs, output_weight(student), y, ht, output_weight(teacher), kl_weight=1.0,
                                      chunk=args.chunk)
                totals["teacher_ce"] += tp.ce
                totals["student_ce"] += sp.ce
                totals["kl"] += sp.kl
                totals["tokens"] += sp.tokens
        n = max(totals.pop("tokens"), 1)
        return {k: v / n for k, v in totals.items()}

    log = open(out / "train.jsonl", "a")
    record = {"event": "start", "args": vars(args), "tokenizer_fingerprint": fingerprint,
              "teacher_trainable": sum(p.numel() for p in teacher_params),
              "student_trainable": sum(p.numel() for p in student.parameters()),
              "eval": evaluate()}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")

    cursor = 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        start = time.time()
        batch = [train[(cursor + i) % len(train)] for i in range(args.accumulate)]
        cursor += args.accumulate
        normaliser = sum(sum(1 for t in labels[1:] if t != IGNORE) for _, labels in batch) or 1
        weight = kl_ramp(step - 1, args.kl_ramp, args.kl_weight)
        sums = {"teacher_ce": 0.0, "student_ce": 0.0, "kl": 0.0, "tokens": 0}
        for ids, labels in batch:
            x, y = shifted(ids, labels, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ht = final_hidden(teacher, x)
                teacher_loss, tp = chunked_ce_kl(ht, output_weight(teacher), y, normaliser=normaliser,
                                                 chunk=args.chunk)
                hs = final_hidden(student, x)
                student_loss, sp = chunked_ce_kl(hs, output_weight(student), y, ht, output_weight(teacher),
                                                 kl_weight=weight, temperature=args.temperature,
                                                 normaliser=normaliser, chunk=args.chunk)
            if teacher_opt is not None:
                teacher_loss.backward()
            student_loss.backward()
            sums["teacher_ce"] += tp.ce
            sums["student_ce"] += sp.ce
            sums["kl"] += sp.kl
            sums["tokens"] += sp.tokens
        student_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
        student_opt.step()
        student_opt.zero_grad(set_to_none=True)
        if teacher_opt is not None:
            torch.nn.utils.clip_grad_norm_(teacher_params, 1.0)
            teacher_opt.step()
            teacher_opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        n = max(sums.pop("tokens"), 1)
        record = {"step": step, **{k: round(v / n, 4) for k, v in sums.items()}, "kl_weight": round(weight, 3),
                  "student_grad_norm": round(student_norm, 3), "supervised_tokens": n,
                  "seconds": round(time.time() - start, 2),
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
    record = {"event": "end", "eval": evaluate()}
    print(json.dumps(record), flush=True)
    log.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
