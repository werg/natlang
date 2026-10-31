#!/usr/bin/env python3
"""Capture one source-bound v8 Luna episode with an already selected student checkpoint.

This tool runs no teacher or judge. Every action is forced from the closed
source audit; three non-admitted decisions are retained with zero target loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))
from natlang_neuralese.data.provider_native_graph import (  # noqa: E402
    capture_luna_native_turn, native_graph_from_luna_episode)
from natlang_neuralese.serve import load_native_engine  # noqa: E402


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    audit_path = Path(args.audit).resolve(strict=True)
    audit = json.loads(audit_path.read_bytes())
    if audit.get("schema") != "natlang.luna-v8-provider-native-ingestion-audit/1":
        raise ValueError("episode capture requires the exact v8 source conversion audit")
    converted_path = Path(audit["python_corpus_conversion"]["path"]).resolve(strict=True)
    converted_bytes = converted_path.read_bytes()
    if sha(converted_bytes) != audit["python_corpus_conversion"]["sha256"]:
        raise ValueError("converted v8 decision file changed after its audit was written")
    raw_path = Path(audit["raw_results"]["path"]).resolve(strict=True)
    raw_bytes = raw_path.read_bytes()
    if sha(raw_bytes) != audit["raw_results"]["sha256"]:
        raise ValueError("raw v8 Luna trajectory changed after its audit was written")
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    rows = [json.loads(line) for line in converted_bytes.splitlines() if line.strip()]
    if len(raw_rows) != 1 or len(rows) != 33 or len(raw_rows[0].get("trajectory") or []) != 33:
        raise ValueError("v8 source audit must bind one exact 33-turn LogHub episode")
    raw = raw_rows[0]
    if raw.get("id") != audit.get("task_id"):
        raise ValueError("v8 episode audit task identity differs from its raw Luna row")
    rows.sort(key=lambda row: row["source_ref"]["provider_input_context"]["source_trajectory_index"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("native episode output directory must be empty to preserve prior captures")
    engine = load_native_engine(args.checkpoint, base=args.base, device=args.device, dtype=dtype)
    engine.start()
    try:
        encoder_cache = {}
        captures = []
        for row in rows:
            index = row["source_ref"]["provider_input_context"]["source_trajectory_index"]
            if index != len(captures):
                raise ValueError("v8 episode decisions are not in exact contiguous source-turn order")
            captures.append(capture_luna_native_turn(engine, str(audit_path), row["id"],
                seed=2650766063 + index, encoder_cache=encoder_cache, allow_context_only=True))
        graph, example = native_graph_from_luna_episode(captures)
        graph_bytes = canonical(graph) + b"\n"
        graph_path = output / "native-episode.jsonl"
        graph_path.write_bytes(graph_bytes)
        receipt = {"schema": "natlang.luna-v8-native-episode-capture/1",
            "audit_path": str(audit_path), "audit_sha256": sha(audit_path.read_bytes()),
            "raw_source_path": str(raw_path), "raw_source_sha256": sha(raw_bytes),
            "converted_decisions_path": str(converted_path),
            "converted_decisions_sha256": sha(converted_bytes),
            "source_turn_count": len(captures),
            "target_supervised_turns": sum(bool(item["source_receipt"]["target_supervision"])
                                             for item in captures),
            "context_only_turns": sum(not item["source_receipt"]["target_supervision"]
                                      for item in captures),
            "supervised_tokens": sum(weight > 0 for weight in example.target_weights),
            "inline_producer_count": len(graph.get("teacher_text_inline_producers") or []),
            "consumer_action_count": sum(bool(item.get("consumer_native_block_ids"))
                                          for item in graph.get("teacher_episode_actions") or []),
            "native_checkpoint_sha256": engine.native_identity.get("checkpoint_sha256"),
            "tokenizer_fingerprint": engine.native_identity.get("tokenizer_fingerprint"),
            "graph_path": str(graph_path), "graph_sha256": sha(graph_bytes),
            "native_graph_invocations": len(example.plan.invocations),
            "source_groups": graph["source_groups"], "oracle_level": "judged",
            "exact_oracle_claimed": False, "training_admission": False}
        receipt_path = output / "receipt.json"
        receipt_path.write_bytes(canonical(receipt) + b"\n")
        print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2))
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
