#!/usr/bin/env python3
"""Build a fresh ordinary gold-text warm-up packet from admitted SFT JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))
from natlang_neuralese.data.text_corpus import gold_text_rows
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_hex as sha  # noqa: E402
from natlang_neuralese.common.jsonio import canonical_json_str as canonical  # noqa: E402


def read_source(path: Path):
    lines = [line for line in path.read_bytes().splitlines() if line]
    rows = [json.loads(line) for line in lines]
    for row, line in zip(rows, lines):
        row["_source_record_sha256"] = sha(line)
    return rows


def token_counts(rows):
    """Token totals per split: whole documents and the supervised assistant suffix (from supervised_suffix_start)."""
    out = {}
    for row in rows:
        split = out.setdefault(row["split"], {"documents": 0, "tokens": 0, "suffix_tokens": 0, "max_document_tokens": 0})
        n = len(row["token_ids"])
        split["documents"] += 1
        split["tokens"] += n
        split["suffix_tokens"] += n - row["supervised_suffix_start"]
        split["max_document_tokens"] = max(split["max_document_tokens"], n)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--pieces", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, help="Exact HF tokenizer ID or local snapshot for the student")
    parser.add_argument("--train-only-addition", action="store_true",
                        help="Render a train-only addition; it supplies no independent evaluation cohort")
    parser.add_argument("--ordinary-text-admission-receipt", type=Path,
                        help="Root approval receipt for a bound ordinary-text-only addition; requires --train-only-addition")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.out}")
    records, pieces = read_source(args.records), [json.loads(line) for line in args.pieces.read_text(encoding="utf-8").splitlines() if line]
    if args.train_only_addition and any(row.get("split") != "train" for row in records):
        parser.error("--train-only-addition requires every source record to retain its train split")
    if args.ordinary_text_admission_receipt and not args.train_only_addition:
        parser.error("--ordinary-text-admission-receipt is scoped to --train-only-addition")
    ordinary_text_receipt = None
    ordinary_text_receipt_sha256 = None
    if args.ordinary_text_admission_receipt:
        ordinary_text_receipt_sha256 = sha(args.ordinary_text_admission_receipt.read_bytes())
        ordinary_text_receipt = json.loads(args.ordinary_text_admission_receipt.read_text(encoding="utf-8"))
    # The pinned-provenance loader: exact tokenizer class and, for TokenizersBackend snapshots, the exact serialized
    # backend (the same guard the delta builder uses); gold_text_rows then binds the renderer fingerprint.
    from build_neuralese_gold_text_delta import load_pinned_tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not (tokenizer_path / "tokenizer.json").is_file():
        parser.error("--tokenizer must be a local tokenizer snapshot directory (pinned provenance)")
    tokenizer = load_pinned_tokenizer(tokenizer_path)
    # A corpus is rendered only under its backbone's declared history-reasoning policy (serve/chat.py
    # BACKBONE_HISTORY_REASONING), asserted against the template; the receipt records it.
    from natlang_neuralese.serve.chat import bind_history_reasoning
    bind_history_reasoning(tokenizer, require_declared=True)
    rows, receipt, omissions, provenance = gold_text_rows(
        records, pieces, tokenizer=tokenizer,
        require_independent_splits=not args.train_only_addition,
        ordinary_text_admission_receipt=ordinary_text_receipt,
        ordinary_text_admission_receipt_sha256=ordinary_text_receipt_sha256)
    if not rows:
        raise ValueError("no renderer-qualified documents in the addition")
    args.out.mkdir(parents=True)
    data = "".join(canonical(row) + "\n" for row in rows).encode("utf-8")
    omission_data = "".join(canonical(row) + "\n" for row in omissions).encode("utf-8")
    provenance_data = "".join(canonical(row) + "\n" for row in provenance).encode("utf-8")
    (args.out / "text.jsonl").write_bytes(data)
    (args.out / "omissions.jsonl").write_bytes(omission_data)
    (args.out / "provenance.jsonl").write_bytes(provenance_data)
    receipt.update({
        "split_scope": "train-only-addition-no-evaluation-credit" if args.train_only_addition else "independent-train-and-held",
        "ordinary_text_admission": ({"schema": "natlang.root-ordinary-text-only-admission/1",
                                     "receipt_sha256": ordinary_text_receipt_sha256,
                                     "approved_rows": len(ordinary_text_receipt.get("rows", [])),
                                     "native_training_admission_granted": False,
                                     "trace_admission_granted": False}
                                    if ordinary_text_receipt is not None else None),
        "source_run": str(args.records.parent),
        "renderer_code": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in (
            Path(__file__).resolve(), ROOT / "training/neuralese/natlang_neuralese/data/text_corpus.py",
            ROOT / "training/neuralese/natlang_neuralese/serve/chat.py")},
        "source_files": {args.records.name: sha(args.records.read_bytes()),
                         args.pieces.name: sha(args.pieces.read_bytes()),
                         **({args.ordinary_text_admission_receipt.name: ordinary_text_receipt_sha256}
                            if args.ordinary_text_admission_receipt else {})},
        "text_jsonl_sha256": sha(data),
        "tokenizer": {"path": str(tokenizer_path.resolve()), "class": type(tokenizer).__name__,
                      "files": {name: sha((tokenizer_path / name).read_bytes()) for name in (
                          "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                          "special_tokens_map.json") if (tokenizer_path / name).is_file()}},
        "tokens": token_counts(rows),
    })
    (args.out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "documents", "train_documents", "test_documents", "omitted_records",
        "excluded_train_exact_held_complete_documents", "text_jsonl_sha256", "tokens")}, indent=2))


if __name__ == "__main__":
    main()
