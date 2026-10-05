#!/usr/bin/env python3
"""Mixed-domain data for Maple QAT and nested-family training (MAPLE_NESTED §4a): broad chat (smoltalk) and text
(fineweb-edu) rendered in Maple's format, as rows like the natlang SFT render ({prompt, completion, source}).

Chat rows: the conversation up to the last assistant turn is the prompt (Maple's chat template, empty think block),
the last assistant turn is the completion. Text rows: empty prompt, the text (truncated) as completion. Broad rows
exist for the anchor KL to the frozen original Maple and for the members' distillation, so general ability is
measured and kept, not only the natlang task.

    python scripts/maple_mixed_data.py --chat .../smoltalk/data/all/train-00000-of-*.parquet \
        --text .../fineweb-edu/sample/10BT/000_00000.parquet --rows 4000 --out runs/maple-nested-20261005/mixed-v1.jsonl
"""

import argparse
import json
import random

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--rows", type=int, default=4000, help="rows of each kind")
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--model", default="/home/werg/data/models/maple-preview-bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    chat = pq.read_table(args.chat).to_pylist()
    random.shuffle(chat)
    text = pq.read_table(args.text, columns=["text", "id"]).to_pylist()
    random.shuffle(text)
    written = {"chat": 0, "text": 0}
    with open(args.out, "w") as out:
        for row in chat:
            if written["chat"] >= args.rows:
                break
            messages = row["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
            completion = messages[-1]["content"] + "<|im_end|>"
            if len(prompt) + len(completion) > args.max_chars:
                continue
            out.write(json.dumps({"id": f"smoltalk:{written['chat']}", "split": "train", "source": "smoltalk",
                                  "prompt": prompt, "completion": completion}) + "\n")
            written["chat"] += 1
        for row in text[:args.rows]:
            out.write(json.dumps({"id": f"fineweb-edu:{row['id']}", "split": "train", "source": "fineweb-edu",
                                  "prompt": "", "completion": row["text"][:args.max_chars]}) + "\n")
            written["text"] += 1
    print(json.dumps(written))


if __name__ == "__main__":
    main()
