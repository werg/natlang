#!/usr/bin/env python3
"""Capture one source-bound v8 Luna episode with an already selected student checkpoint.

This tool runs no teacher or judge. Every action is forced from the closed
source audit; three non-admitted decisions are retained with zero target loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))
from natlang_neuralese.data.provider_native_graph import (  # noqa: E402
    capture_luna_native_turn, native_graph_from_luna_episode)
from natlang_neuralese.serve import load_native_engine  # noqa: E402
from natlang_neuralese.common.hashing import sha256_hex  # noqa: E402
from natlang_neuralese.common.jsonio import canonical_json_bytes  # noqa: E402


class _HttpNativeEngine:
    """Use an already resident actor without loading a second checkpoint."""
    def __init__(self, endpoint):
        self.endpoint = endpoint.rstrip("/")
        info = self._request("GET", "/v1/neuralese/info")
        state = info.get("native_state") or {}
        self.native_identity = {key: state.get(key) for key in
            ("checkpoint_sha256", "tokenizer_fingerprint", "embedding_weight_sha256", "architecture")}
        if (info.get("server") != "reference" or state.get("enabled") is not True
                or not self.native_identity.get("checkpoint_sha256")
                or not self.native_identity.get("tokenizer_fingerprint")):
            raise ValueError("endpoint is not a ready reference native-state actor")
        self.store = self

    def _request(self, method, path, body=None, *, binary=False):
        data = None if body is None else (body if isinstance(body, bytes) else canonical_json_bytes(body))
        request = urllib.request.Request(self.endpoint + path, data=data, method=method,
            headers={"Content-Type": "application/octet-stream" if binary else "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"native actor {method} {path} returned HTTP {error.code}: {detail}") from error
        return raw if binary else json.loads(raw)

    def start(self):
        return None

    def stop(self):
        return None

    def native_write_text(self, text, *, value_type=None):
        return self._request("POST", "/v1/neuralese/native-states/write-text",
            {"text": text, "type": value_type})

    def generate(self, request):
        return self._request("POST", "/v1/chat/completions", {
            "messages": request.messages, "tools": request.tools,
            "max_tokens": request.max_tokens, "temperature": request.temperature,
            "seed": request.seed, "x_natlang_rl_capture": True,
            "x_natlang_native_recording": True,
            "x_natlang_forced_message": request.forced_message,
        })

    def get_native_state(self, state_id):
        from natlang_neuralese.serve.store import decode_native_state
        raw = self._request("GET", f"/v1/neuralese/native-states/{state_id}", binary=True)
        block, actual_id = decode_native_state(raw)
        if actual_id != state_id:
            raise ValueError("native actor returned a state with a mismatched content address")
        return SimpleNamespace(id=actual_id, block=block)

    def metrics(self):
        raw = self._request("GET", "/metrics", binary=True)
        return {key: int(value) for line in raw.decode().splitlines()
                if len(parts := line.split()) == 2
                and parts[0] in {"natlang_native_decode_tokens_total", "natlang_active_sequences",
                                "natlang_queued_requests", "natlang_native_open_produce_spans"}
                for key, value in (parts,)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--endpoint", help="Reuse an already running native reference actor over HTTP")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if bool(args.endpoint) == bool(args.checkpoint):
        parser.error("supply exactly one of --endpoint or --checkpoint")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    audit_path = Path(args.audit).resolve(strict=True)
    audit = json.loads(audit_path.read_bytes())
    if audit.get("schema") != "natlang.luna-v8-provider-native-ingestion-audit/1":
        raise ValueError("episode capture requires the exact v8 source conversion audit")
    converted_path = Path(audit["python_corpus_conversion"]["path"]).resolve(strict=True)
    converted_bytes = converted_path.read_bytes()
    if sha256_hex(converted_bytes) != audit["python_corpus_conversion"]["sha256"]:
        raise ValueError("converted v8 decision file changed after its audit was written")
    raw_path = Path(audit["raw_results"]["path"]).resolve(strict=True)
    raw_bytes = raw_path.read_bytes()
    if sha256_hex(raw_bytes) != audit["raw_results"]["sha256"]:
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
    engine = (_HttpNativeEngine(args.endpoint) if args.endpoint else
        load_native_engine(args.checkpoint, base=args.base, device=args.device, dtype=dtype))
    if args.endpoint and engine.native_identity.get("checkpoint_sha256") is None:
        raise ValueError("native endpoint did not report a checkpoint identity")
    engine.start()
    try:
        encoder_cache = {}
        captures = []
        for row in rows:
            index = row["source_ref"]["provider_input_context"]["source_trajectory_index"]
            if index != len(captures):
                raise ValueError("v8 episode decisions are not in exact contiguous source-turn order")
            started = time.monotonic()
            before = engine.metrics() if args.endpoint else {}
            print(json.dumps({"event": "native_turn_start", "source_index": index,
                "record_id": row["id"], "source_prompt_json_bytes": len(canonical_json_bytes(row["messages"])),
                "source_prompt_message_count": len(row["messages"]), "decode_tokens_before":
                    before.get("natlang_native_decode_tokens_total")}), flush=True)
            try:
                captured = capture_luna_native_turn(engine, str(audit_path), row["id"],
                    seed=2650766063 + index, encoder_cache=encoder_cache, allow_context_only=True)
            except Exception as error:
                after = engine.metrics() if args.endpoint else {}
                print(json.dumps({"event": "native_turn_error", "source_index": index,
                    "record_id": row["id"], "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": str(error), "metrics_before": before, "metrics_after": after}), flush=True)
                raise
            captures.append(captured)
            after = engine.metrics() if args.endpoint else {}
            usage = captured["response"].get("usage") or {}
            print(json.dumps({"event": "native_turn_complete", "source_index": index,
                "record_id": row["id"], "elapsed_seconds": round(time.monotonic() - started, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "target_supervision": captured["source_receipt"]["target_supervision"],
                "native_writes": len(captured["capture"].get("writes") or []),
                "native_imports": sum(segment.get("kind") == "native_state"
                    for segment in captured["capture"].get("prompt") or []),
                "metrics_before": before, "metrics_after": after}), flush=True)
        graph, example = native_graph_from_luna_episode(captures)
        graph_bytes = canonical_json_bytes(graph) + b"\n"
        graph_path = output / "native-episode.jsonl"
        graph_path.write_bytes(graph_bytes)
        receipt = {"schema": "natlang.luna-v8-native-episode-capture/1",
            "audit_path": str(audit_path), "audit_sha256": sha256_hex(audit_path.read_bytes()),
            "raw_source_path": str(raw_path), "raw_source_sha256": sha256_hex(raw_bytes),
            "converted_decisions_path": str(converted_path),
            "converted_decisions_sha256": sha256_hex(converted_bytes),
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
            "actor_endpoint": args.endpoint,
            "tokenizer_fingerprint": engine.native_identity.get("tokenizer_fingerprint"),
            "graph_path": str(graph_path), "graph_sha256": sha256_hex(graph_bytes),
            "native_graph_invocations": len(example.plan.invocations),
            "source_groups": graph["source_groups"], "oracle_level": "judged",
            "exact_oracle_claimed": False, "training_admission": False}
        receipt_path = output / "receipt.json"
        receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
        print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2))
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
