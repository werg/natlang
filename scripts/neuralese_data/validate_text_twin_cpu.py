#!/usr/bin/env python3
"""Load a tokenizer-twin text file with the current shared trainer loader (CPU) and report its shape.

Usage: validate_text_twin_cpu.py TWIN_TEXT SOURCE_TEXT TOKENIZER_DIR [--context N ...]
Prints counts, token length stats and, for each N, the fraction of supervised tokens whose
preceding context exceeds N tokens (twin and source).
"""
import argparse, hashlib, json, os, platform, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training/neuralese"))
from transformers import AutoTokenizer
import transformers, torch
from natlang_neuralese.train.text_warmup import load_text_rows
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def shape(rows, contexts):
    lens = sorted(len(r["token_ids"]) for r in rows)
    sup = sum(len(r["token_ids"]) - r["supervised_suffix_start"] for r in rows)
    over = {}
    for n in contexts:
        # supervised token at position p is predicted from the p tokens before it
        lost = sum(max(0, len(r["token_ids"]) - max(r["supervised_suffix_start"], n + 1)) for r in rows)
        over[str(n)] = {"supervised_tokens_with_context_over": lost, "fraction": lost / sup,
                        "documents_with_any": sum(len(r["token_ids"]) > max(r["supervised_suffix_start"], n + 1)
                                                   for r in rows)}
    q = lambda f: lens[min(len(lens) - 1, int(f * len(lens)))]
    return {"documents": len(rows), "tokens": sum(lens), "supervised_tokens": sup,
            "length": {"mean": sum(lens) / len(lens), "median": statistics.median(lens), "p90": q(0.9), "max": lens[-1]},
            "context_over": over}

ap = argparse.ArgumentParser()
ap.add_argument("twin"); ap.add_argument("source"); ap.add_argument("tokenizer")
ap.add_argument("--context", type=int, nargs="*", default=[7168, 31744])
a = ap.parse_args()
before = sha(a.twin)
tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
rows, receipt = load_text_rows(None, None, a.twin, tokenizer=tok)
fp = tokenizer_fingerprint(tok)
src = [json.loads(l) for l in open(a.source)]
counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "test")}
groups = {s: {g for r in rows if r["split"] == s for g in r["source_groups"]} for s in ("train", "test")}
result = {"status": "passed shared load_text_rows validation", "twin_sha256": before, "loader_receipt": receipt,
          "rows": len(rows), "split_counts": counts, "unique_ids": len({r["id"] for r in rows}),
          "unique_source_record_ids": len({s for r in rows for s in r["source_record_ids"]}),
          "train_test_source_group_intersection": len(groups["train"] & groups["test"]),
          "tokenizer_fingerprint": fp,
          "rows_fingerprint_match": all(r.get("tokenizer_sha256") == fp for r in rows),
          "all_rows_targetbound": all(type(r.get("supervised_suffix_start")) is int and 0 < r["supervised_suffix_start"] < len(r["token_ids"]) for r in rows),
          "twin": {s: shape([r for r in rows if r["split"] == s], a.context) for s in ("train", "test")} | {"all": shape(rows, a.context)},
          "source": shape(src, a.context),
          "sha256_unchanged": sha(a.twin) == before, "cuda_available": torch.cuda.is_available(),
          "transformers_version": transformers.__version__, "python_version": platform.python_version()}
ok = (result["rows_fingerprint_match"] and result["all_rows_targetbound"] and result["sha256_unchanged"]
      and result["train_test_source_group_intersection"] == 0)
if not ok:
    result["status"] = "FAILED"
print(json.dumps(result, sort_keys=True, indent=1))
sys.exit(0 if ok else 1)
