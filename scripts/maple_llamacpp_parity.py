#!/usr/bin/env python3
"""maple-qat M0.3: our PyTorch Maple against llama.cpp on the official GGUF, by perplexity on the same chunks.

llama-perplexity splits the token stream into chunks of n_ctx tokens and scores the second half of each chunk.
This script writes a text file from corpus rows, runs llama-perplexity on it, and reproduces the same chunks
with our model (``natlang_neuralese.maple.model``), reporting per-chunk NLL from both. With n_ctx > 512 the
sliding-window rule is exercised.

    python scripts/maple_llamacpp_parity.py --gguf ... --checkpoint /home/werg/data/models/maple-preview-bf16 \
        --data runs/maple-joint-20261005/qwen3-render-v1.jsonl --out runs/maple-nested-20261005/m0.3
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))
PERPLEXITY = Path("/home/werg/llama.cpp-neuralese/build-cpu/bin/llama-perplexity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from natlang_neuralese.maple.model import load_maple

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.data)][:args.rows]
    text = "\n\n".join(r["completion"] + "\n" + r["prompt"][-4000:] for r in rows)
    (out / "text.txt").write_text(text)

    command = [str(PERPLEXITY), "-m", args.gguf, "-f", str(out / "text.txt"), "-c", str(args.ctx), "-b",
               str(args.ctx), "--chunks", str(args.chunks), "-t", str(args.threads)]
    run = subprocess.run(command, capture_output=True, text=True)
    (out / "llama-perplexity.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise SystemExit(run.stderr[-2000:])
    cumulative = [float(x) for x in re.findall(r"\[\d+\]([0-9.]+)", run.stdout + run.stderr)]
    final = re.search(r"Final estimate: PPL = ([0-9.]+)", run.stdout + run.stderr)
    add_bos = "add_bos_token = true" in (run.stdout + run.stderr) or "tokenizer.ggml.add_bos_token bool = true" in (run.stdout + run.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    model = load_maple(args.checkpoint, device="cuda")
    head = model.get_output_embeddings().weight.float()
    total, count, ours = 0.0, 0, []
    with torch.no_grad():
        for c in range(args.chunks):
            chunk = tokens[c * args.ctx:(c + 1) * args.ctx]
            if len(chunk) < args.ctx:
                break
            if add_bos:
                chunk = [tokenizer.bos_token_id] + chunk[1:]
            ids = torch.tensor([chunk], device="cuda")
            h = model.model(input_ids=ids).last_hidden_state[0].float()
            first = args.ctx // 2
            logits = h[first:-1] @ head.T
            nll = torch.nn.functional.cross_entropy(logits, ids[0, first + 1:], reduction="sum").item()
            total += nll
            count += args.ctx - first - 1
            ours.append(math.exp(total / count))
    report = {"schema": "natlang.maple_llamacpp_parity/1", "gguf": args.gguf, "ctx": args.ctx,
              "llama_cpp_cumulative_ppl": cumulative, "llama_cpp_final": float(final.group(1)) if final else None,
              "ours_cumulative_ppl": ours, "add_bos": add_bos, "tokens": len(tokens)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
