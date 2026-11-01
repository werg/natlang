#!/usr/bin/env python3
"""Apply an exact, reviewed token-target mask to complete native graph rows.

The request binds the immutable input JSONL, each graph's semantic binding,
and the exact token ranges to zero. The output retains every graph node and
edge; this is a supervision change, not a trajectory rewrite.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_graph_quality import mask_exact_target_spans
from natlang_neuralese.data.native_graph_schema import parse_graph


def _bind(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path),
            "bytes": path.stat().st_size}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid graph JSON at line {line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"graph JSON at line {line_no} is not an object")
        rows.append(row)
    return rows


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build(request_path: str | Path, output_path: str | Path,
          receipt_path: str | Path) -> dict:
    request_path = Path(request_path).resolve(strict=True)
    output_path, receipt_path = Path(output_path), Path(receipt_path)
    if output_path.exists() or receipt_path.exists():
        raise ValueError("mask outputs must be new paths; existing evidence is immutable")
    request_binding = _bind(request_path)
    request = json.loads(request_path.read_bytes())
    if request.get("schema") != "natlang.native-graph-target-mask-request/1":
        raise ValueError("unsupported exact target-mask request schema")
    review = request.get("review")
    if (not isinstance(review, dict) or not isinstance(review.get("review_id"), str)
            or not review["review_id"].strip() or not isinstance(review.get("rationale"), str)
            or not review["rationale"].strip()):
        raise ValueError("request needs a named review and non-empty rationale")
    source = request.get("source") or {}
    source_path = Path(source.get("path", "")).resolve(strict=True)
    source_binding = _bind(source_path)
    if source_binding["sha256"] != source.get("sha256"):
        raise ValueError("source JSONL bytes differ from the reviewed request")
    tokenizer = request.get("tokenizer_fingerprint")
    targets = request.get("targets")
    if not isinstance(tokenizer, str) or len(tokenizer) != 64 or not isinstance(targets, list) or not targets:
        raise ValueError("request needs a tokenizer fingerprint and exact target declarations")
    rows = _read_jsonl(source_path)
    positions: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        ident = row.get("id")
        if isinstance(ident, str):
            positions.setdefault(ident, []).append(i)
    requested_ids = [target.get("id") if isinstance(target, dict) else None for target in targets]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("each graph may be reviewed at most once")
    if set(requested_ids) - set(positions):
        raise ValueError("a reviewed graph ID is absent from the input source")
    if any(len(positions[ident]) != 1 for ident in requested_ids):
        raise ValueError("a reviewed graph ID is duplicated in the input source")
    transforms = []
    for target in targets:
        row = rows[positions[target["id"]][0]]
        parsed = parse_graph(row, tokenizer_fingerprint=tokenizer)
        if parsed.split != "train":
            raise ValueError("exact target masking is restricted to train graphs")
        changed, transform_receipt = mask_exact_target_spans(
            row, tokenizer_fingerprint=tokenizer,
            expected_id=target["id"], expected_binding=target.get("binding", ""),
            spans=target.get("spans", []))
        rows[positions[target["id"]][0]] = changed
        transforms.append(transform_receipt)

    payload = ("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                     for row in rows)).encode()
    _write_exclusive(output_path, payload)
    output_binding = _bind(output_path)
    code_paths = [Path(__file__).resolve(),
                  ROOT / "training/neuralese/natlang_neuralese/data/native_graph_quality.py",
                  ROOT / "training/neuralese/natlang_neuralese/data/native_graph_schema.py"]
    receipt = {
        "schema": "natlang.native-graph-target-mask-build/1",
        "request": request_binding,
        "source": source_binding,
        "output": output_binding,
        "review": review,
        "tokenizer_fingerprint": tokenizer,
        "source_rows": len(rows),
        "masked_graphs": transforms,
        "mask_implementation": [_bind(p) for p in code_paths],
        "training_admission": False,
        "limitations": ["Complete graph topology and forward tokens are preserved; downstream admission remains separate."]}
    write_json_durable(receipt_path, receipt)
    return {"receipt": _bind(receipt_path), **receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.request, args.output_jsonl, args.receipt), indent=2))


if __name__ == "__main__":
    main()
