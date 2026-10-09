#!/usr/bin/env python3
"""Render an explicitly held native-action set through the shared gold renderer."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_hex as sha  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))


def read_records(path: Path):
    rows = []
    for line in path.open("rb"):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_source_record_sha256"] = sha(line.rstrip(b"\r\n"))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--pieces", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("fresh or empty output directory required")
    from natlang_neuralese.data.text_corpus import gold_text_preview_rows, tokenizer_fingerprint
    from transformers import AutoTokenizer

    records = read_records(args.records)
    pieces = [json.loads(line) for line in args.pieces.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    rows, receipt, omissions, provenance = gold_text_preview_rows(records, pieces, tokenizer=tokenizer)
    if receipt.get("review_only") is not True or receipt.get("sft_eligible") is not False:
        raise ValueError("shared renderer did not return a held-only receipt")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (("text.jsonl", rows), ("omissions.jsonl", omissions), ("provenance.jsonl", provenance)):
        data = "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                       for value in values).encode("utf-8")
        (args.out / name).write_bytes(data)
        receipt.setdefault("output_sha256", {})[name] = sha(data)
    receipt["input_sha256"] = {
        args.records.name: sha(args.records.read_bytes()),
        args.pieces.name: sha(args.pieces.read_bytes()),
    }
    receipt["renderer_code_sha256"] = sha((ROOT / "training/neuralese/natlang_neuralese/data/text_corpus.py").read_bytes())
    receipt["tokenizer_path"] = str(args.tokenizer)
    receipt["tokenizer_sha256"] = tokenizer_fingerprint(tokenizer)
    receipt["preview_api"] = "gold_text_preview_rows"
    (args.out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "format", "status", "review_only", "sft_eligible", "documents", "train_documents",
        "test_documents", "omitted_records", "tokenizer_sha256", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
