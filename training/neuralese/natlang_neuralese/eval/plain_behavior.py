"""Paired greedy text-behavior comparison for exact LFM backbone checkpoints.

Each role loads the same pinned base and tokenizer, renders the same pinned
chat probes with that tokenizer's own template, and calls the ordinary HF
causal-LM generator. The runner does not install Neuralese heads into decoding,
sample, repair outputs, or treat this descriptive report as a qualification.

Run with ``python -m natlang_neuralese.eval.plain_behavior``. Probe JSONL rows
use the shared generation-check fields (id, messages, optional tools,
enable_thinking, kind, expected_tool, legacy numeric answer, or the stricter
offline expected_exact_text). Legacy numeric answer presence is only structural.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import importlib.metadata
import json
import os
from pathlib import Path

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from ..serve.chat import normalize_tool_call_arguments
from .generation_checks import check, summarize


def _execution_components():
    """Pin every in-package Python source module used by loading and evaluation."""
    package = Path(__file__).resolve().parents[1]
    return {p.relative_to(package).as_posix(): sha256_file_hex(p)
            for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts}


def _append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_status(path: Path, status: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _dependency_versions():
    names = ("torch", "transformers", "tokenizers", "peft", "huggingface-hub", "safetensors")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _render(tokenizer, probe):
    normalized_messages, conversions = normalize_tool_call_arguments(probe["messages"])
    encoded = tokenizer.apply_chat_template(
        normalized_messages, tools=probe.get("tools") or None, tokenize=True,
        add_generation_prompt=True, enable_thinking=bool(probe.get("enable_thinking")),
        return_tensors="pt")
    return encoded, normalized_messages, conversions


def _load_model(checkpoint: Path | None, *, device: str, base: str | None):
    if checkpoint is None:
        from ..model.lfm2_port import load_backbone
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model, tokenizer = load_backbone(base=base, dtype=dtype, device=device)
        return model, tokenizer
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    # Both the older recurrence checkpoint and current full-head exports pass
    # through the shared strict checkpoint loader. Decoding below bypasses its
    # Neuralese Engine.generate path and calls the restored HF backbone directly.
    from ..serve import load_engine
    engine = load_engine(base=base, heads_checkpoint=str(checkpoint), device=device, dtype=dtype)
    # Deliberately bypass Engine.generate and its Neuralese state machine. This
    # comparison asks what the ordinary LFM causal LM emits from plain text.
    return engine.backbone.hf, engine.tokenizer


@torch.inference_mode()
def _run_role(role: str, checkpoint: Path | None, probes: list[dict], *, device: str,
              base: str | None, max_new_tokens: int, journal: Path,
              status_path: Path, progress: dict) -> list[dict]:
    model, tokenizer = _load_model(checkpoint, device=device, base=base)
    model.eval()
    device_obj = torch.device(device)
    end_ids = {int(i) for i in (tokenizer.eos_token_id,
              tokenizer.convert_tokens_to_ids("<|im_end|>"))
               if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id}
    if not end_ids:
        raise ValueError("the pinned LFM tokenizer must expose eos_token_id or <|im_end|>")
    _append_jsonl(journal, {"event": "role_loaded", "role": role,
                            "model_name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None),
                            "model_commit_hash": getattr(getattr(model, "config", None), "_commit_hash", None),
                            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
                            "tokenizer_commit_hash": (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash"),
                            "stop_token_ids": sorted(end_ids)})
    results = []
    for probe in probes:
        encoded, normalized_messages, argument_conversions = _render(tokenizer, probe)
        # Transformers 5 returns BatchEncoding, a Mapping, from
        # apply_chat_template(return_tensors="pt"); older versions may return
        # the input-id tensor directly. Normalize either declared shape here.
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("tokenizer chat template must return an input_ids Tensor or mapping")
        if encoded.ndim == 1:
            encoded = encoded[None, :]
        prompt_len = int(encoded.shape[-1])
        generated = model.generate(
            input_ids=encoded.to(device_obj), do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens, use_cache=True,
            eos_token_id=sorted(end_ids), pad_token_id=(tokenizer.pad_token_id
                                                        if tokenizer.pad_token_id is not None else min(end_ids)))
        output_ids = generated[0, prompt_len:].detach().cpu().tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=False)
        checks = check(probe, text, output_ids, sorted(end_ids))
        prompt_record = {"messages": probe["messages"],
                         "normalized_messages": normalized_messages,
                         "tool_argument_normalizations": argument_conversions,
                         "tools": probe.get("tools"),
                         "enable_thinking": bool(probe.get("enable_thinking")),
                         "rendered_input_ids": encoded[0].detach().cpu().tolist()}
        results.append({"role": role, "id": probe["id"], "kind": probe.get("kind"),
                        "output_convention": probe.get("output_convention", "maple"),
                        "prompt_sha256": canonical_json_sha256_hex(prompt_record),
                        "tool_argument_normalizations": argument_conversions,
                        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
                        "model_name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None),
                        "stop_token_ids": sorted(end_ids),
                        "prompt_tokens": prompt_len, "completion_tokens": len(output_ids),
                        "completion_token_ids": output_ids, "completion_text": text,
                        "checks": checks, "pass": all(checks.values())})
        _append_jsonl(journal, {"event": "probe_completed", "role": role, "result": results[-1]})
        progress["completed"] += 1
        progress["current_role_completed"] += 1
        _write_status(status_path, {"schema": "natlang.plain-text-behavior-run-status/1",
                                    "status": "running", "selected_probe_count": len(probes),
                                    "completed_roles": progress["completed_roles"],
                                    "completed_role_probes": progress["completed"],
                                    "current_role": role,
                                    "current_role_completed": progress["current_role_completed"]})
    del model
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _role_summary(results):
    summary = summarize(results)
    summary["probe_count"] = len(results)
    summary["completion_tokens"] = sum(row["completion_tokens"] for row in results)
    return summary


def evaluate(args):
    if args.out.exists():
        raise ValueError("output directory must be new; preserve every paired run")
    if sha256_file_hex(args.probes) != args.probes_sha256:
        raise ValueError("probe file changed from its declared SHA-256")
    probes = [json.loads(line) for line in args.probes.read_text().splitlines() if line.strip()]
    if not probes or any(not isinstance(p, dict) or not isinstance(p.get("id"), str) or
                         not isinstance(p.get("messages"), list) for p in probes):
        raise ValueError("probe JSONL requires at least one row with string id and messages array")
    if len({p["id"] for p in probes}) != len(probes):
        raise ValueError("probe IDs must be unique")
    selected = probes[:args.limit] if args.limit else probes
    if not selected:
        raise ValueError("positive --limit required when provided")
    if args.base is None:
        from ..model.lfm2_port import DEFAULT_BASE, DEFAULT_REVISION
        base, base_revision = DEFAULT_BASE, DEFAULT_REVISION
    else:
        base, base_revision = args.base, None
    execution_pins = _execution_components()
    if not args.candidate_state_checkpoint.is_file() or not args.candidate_receipt.is_file():
        raise FileNotFoundError("candidate full-state checkpoint and best-checkpoint receipt are required")
    if not args.base_snapshot.is_dir():
        raise FileNotFoundError("a local pinned base snapshot is required for the read-only weight-drift comparison")
    receipt = json.loads(args.candidate_receipt.read_text())
    state_hash = sha256_file_hex(args.candidate_state_checkpoint)
    candidate_hash = sha256_file_hex(args.candidate_checkpoint)
    files = receipt.get("files", {})
    state_pin = files.get("best-checkpoint.pt", {}).get("sha256")
    candidate_pin = files.get("best-heads.pt", {}).get("sha256")
    if (receipt.get("schema") != "natlang.neuralese-best-warmup-checkpoint/1" or
            state_pin != state_hash or candidate_pin != candidate_hash):
        raise ValueError("candidate files must be the exact best-checkpoint/best-heads pair in the receipt")
    # Header-only mmap reads bind the candidate serving export to that full
    # selected state without materializing a second model or touching GPU.
    state_header = torch.load(args.candidate_state_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    export_header = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state_step = int(state_header.get("step", -1))
    export_step = int((export_header.get("warmup") or {}).get("step", -2))
    if state_step != int(receipt.get("step", -3)) or export_step != state_step:
        raise ValueError("candidate export and full-state checkpoint are not the exact selected step")
    candidate_structure = {"lora_tensor_count": len(export_header.get("lora") or {}),
                           "backbone_training": export_header.get("backbone_training"),
                           "backbone_trainable_tensor_count": len(export_header.get("backbone_trainables") or {})}
    if candidate_structure["lora_tensor_count"] == 0 or candidate_structure["backbone_training"] != "full" or candidate_structure["backbone_trainable_tensor_count"] == 0:
        raise ValueError("candidate export lacks the expected LoRA and full-backbone trainable tensors")
    del state_header
    lineage_header = torch.load(args.lineage_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    lineage_schema = lineage_header.get("schema")
    lineage_lora = lineage_header.get("lora") or {}
    lineage_lora_count = len(lineage_lora)
    lineage_identity = lineage_header.get("identity") or {}
    lineage_rank = (lineage_identity.get("options") or {}).get("rank")
    lineage_layers = sorted({int(name.split("model.layers.", 1)[1].split(".", 1)[0])
                             for name in lineage_lora if "model.layers." in name})
    if lineage_schema != "natlang.neuralese_recurrence_checkpoint/1" or lineage_lora_count == 0:
        raise ValueError("lineage checkpoint has no embedded LoRA state for shared load_engine restoration")
    if not isinstance(lineage_rank, int) or not lineage_layers:
        raise ValueError("lineage checkpoint must declare LoRA rank and identify its exact adapter layer set")
    del lineage_header, lineage_lora, export_header
    args.out.mkdir(parents=True)
    journal = args.out / "execution-journal.jsonl"
    status_path = args.out / "run-status.json"
    _write_status(status_path, {"schema": "natlang.plain-text-behavior-run-status/1",
                                                   "status": "running", "selected_probe_count": len(selected),
                                                   "completed_role_probes": 0})
    invocation = {"schema": "natlang.plain-text-behavior-invocation/1",
                  "probe_path": str(args.probes), "probe_sha256": args.probes_sha256,
                  "selected_probe_ids": [p["id"] for p in selected],
                  "lineage_checkpoint": str(args.lineage_checkpoint),
                  "lineage_checkpoint_sha256": sha256_file_hex(args.lineage_checkpoint),
                  "candidate_checkpoint": str(args.candidate_checkpoint),
                  "candidate_checkpoint_sha256": candidate_hash,
                  "candidate_state_checkpoint": str(args.candidate_state_checkpoint),
                  "candidate_state_checkpoint_sha256": state_hash,
                  "candidate_selection_receipt": str(args.candidate_receipt),
                  "candidate_selection_receipt_sha256": sha256_file_hex(args.candidate_receipt),
                  "base": base, "base_revision": base_revision,
                  "device": args.device, "max_new_tokens": args.max_new_tokens,
                  "execution_components": execution_pins}
    invocation["lineage_checkpoint_structure"] = {"schema": lineage_schema,
                                                   "lora_tensor_count": lineage_lora_count,
                                                   "rank": lineage_rank, "layers": lineage_layers}
    (args.out / "invocation.json").write_text(json.dumps(invocation, indent=2, ensure_ascii=False) + "\n")
    with (args.out / "invocation.json").open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    _append_jsonl(journal, {"event": "run_started", "probe_sha256": args.probes_sha256,
                            "execution_components": execution_pins,
                            "lineage_checkpoint_sha256": sha256_file_hex(args.lineage_checkpoint),
                            "lineage_lora_tensor_count": lineage_lora_count,
                            "lineage_lora_rank": lineage_rank,
                            "lineage_lora_layers": lineage_layers,
                            "lineage_identity_keys": sorted(lineage_identity),
                            "candidate_structure": candidate_structure})
    roles = [("unmodified_pinned_base", None),
             ("original_pre_alignment_student", args.lineage_checkpoint),
             ("selected_candidate", args.candidate_checkpoint)]
    role_results = {}
    progress = {"completed": 0, "completed_roles": [], "current_role_completed": 0}
    try:
        for name, checkpoint in roles:
            if checkpoint is not None and not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            _append_jsonl(journal, {"event": "role_started", "role": name,
                                    "checkpoint_sha256": sha256_file_hex(checkpoint) if checkpoint else None})
            progress["current_role_completed"] = 0
            role_results[name] = _run_role(name, checkpoint, selected, device=args.device,
                                           base=args.base, max_new_tokens=args.max_new_tokens,
                                           journal=journal, status_path=status_path,
                                           progress=progress)
            _append_jsonl(journal, {"event": "role_completed", "role": name,
                                    "probe_count": len(role_results[name])})
            progress["completed_roles"].append(name)
    except BaseException as error:
        _append_jsonl(journal, {"event": "run_error", "error_type": type(error).__name__,
                                "error": str(error)[:1000]})
        _write_status(args.out / "run-status.json", {"schema": "natlang.plain-text-behavior-run-status/1",
                                                      "status": "failed", "error_type": type(error).__name__,
                                                      "error": str(error)[:1000]})
        raise
    _append_jsonl(journal, {"event": "weight_drift_started", "base_snapshot": str(args.base_snapshot)})
    try:
        from .plain_weight_drift import compare_checkpoint_weights
        drift = compare_checkpoint_weights(args.lineage_checkpoint, args.candidate_checkpoint, args.base_snapshot)
        drift_path = args.out / "weight-drift.json"
        temporary = drift_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(drift, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, drift_path)
        _append_jsonl(journal, {"event": "weight_drift_completed",
                                "compared_tensor_count": drift["compared_tensor_count"],
                                "aggregate_relative_l2": drift["aggregate_relative_l2"],
                                "weight_drift_sha256": sha256_file_hex(drift_path)})
    except BaseException as error:
        _append_jsonl(journal, {"event": "run_error", "error_type": type(error).__name__,
                                "error": str(error)[:1000]})
        _write_status(status_path, {"schema": "natlang.plain-text-behavior-run-status/1",
                                    "status": "failed", "error_type": type(error).__name__,
                                    "error": str(error)[:1000]})
        raise
    if _execution_components() != execution_pins:
        _append_jsonl(journal, {"event": "run_error", "error_type": "ExecutionCodeDrift",
                                "error": "Python package source changed during paired execution or drift comparison"})
        _write_status(status_path, {"schema": "natlang.plain-text-behavior-run-status/1",
                                    "status": "failed", "error_type": "ExecutionCodeDrift"})
        raise RuntimeError("paired evaluation code changed during execution; preserve outputs and rerun under a frozen package")
    paired = []
    try:
        for i, probe in enumerate(selected):
            rows = {name: role_results[name][i] for name, _ in roles}
            if (len({row["prompt_sha256"] for row in rows.values()}) != 1 or
                    len({tuple(row["stop_token_ids"]) for row in rows.values()}) != 1):
                raise AssertionError("paired roles used different rendered prompts or stop-token IDs")
            paired.append({"id": probe["id"], "kind": probe.get("kind"),
                           "prompt_sha256": rows["unmodified_pinned_base"]["prompt_sha256"],
                           "roles": rows})
    except BaseException as error:
        _append_jsonl(journal, {"event": "run_error", "error_type": type(error).__name__,
                                "error": str(error)[:1000]})
        _write_status(status_path, {"schema": "natlang.plain-text-behavior-run-status/1",
                                    "status": "failed", "error_type": type(error).__name__,
                                    "error": str(error)[:1000]})
        raise
    with (args.out / "paired-results.jsonl").open("x") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"schema": "natlang.plain-text-behavior-comparison/1",
               "interpretation": "descriptive paired generation comparison only; no qualification or admission",
               "inputs": {"probes": str(args.probes), "probes_sha256": args.probes_sha256,
                          "selected_probe_ids": [p["id"] for p in selected],
                          "lineage_checkpoint": str(args.lineage_checkpoint),
                          "lineage_checkpoint_sha256": sha256_file_hex(args.lineage_checkpoint),
                          "candidate_checkpoint": str(args.candidate_checkpoint),
                          "candidate_checkpoint_sha256": candidate_hash,
                          "candidate_state_checkpoint": str(args.candidate_state_checkpoint),
                          "candidate_state_checkpoint_sha256": state_hash,
                          "candidate_selection_receipt": str(args.candidate_receipt),
                          "candidate_selection_receipt_sha256": sha256_file_hex(args.candidate_receipt),
                          "base": base, "base_revision": base_revision,
                          "base_snapshot": str(args.base_snapshot),
                          "base_model_weights_sha256": drift["base_weights_sha256"]},
               "decoding": {"method": "greedy", "do_sample": False, "num_beams": 1,
                            "max_new_tokens": args.max_new_tokens,
                            "stop_token_ids": "tokenizer eos_token_id plus <|im_end|> when present",
                            "template": "the pinned base tokenizer apply_chat_template with add_generation_prompt"},
               "execution": {"components": execution_pins,
                             "source_coverage": "all natlang_neuralese Python modules; external dependencies are version-pinned, not transitively vendored",
                             "dependencies": _dependency_versions()},
               "roles": {name: _role_summary(rows) for name, rows in role_results.items()},
               "weight_drift": {"path": str(args.out / "weight-drift.json"),
                                "sha256": sha256_file_hex(args.out / "weight-drift.json"),
                                "aggregate_relative_l2": drift["aggregate_relative_l2"],
                                "compared_tensor_count": drift["compared_tensor_count"],
                                "interpretation": drift["interpretation"]},
               "comparison": {"completion_texts_saved_per_role": True,
                              "same_prompt_hash_all_roles": True,
                              "automatic_gate_effect": False,
                              "legacy_numeric_answer_check": "expected digit presence only; structural sanity signal, not semantic correctness",
                              "exact_plain_answer_check": "expected_exact_text compares trimmed final response with an offline expected string"}}
    summary_path = args.out / "summary.json"
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, summary_path)
    _write_status(args.out / "run-status.json", {"schema": "natlang.plain-text-behavior-run-status/1",
                                                   "status": "complete", "selected_probe_count": len(selected),
                                                   "completed_roles": list(role_results),
                                                   "completed_role_probes": sum(map(len, role_results.values())),
                                                   "summary_sha256": sha256_file_hex(summary_path)})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--probes-sha256", required=True)
    parser.add_argument("--lineage-checkpoint", type=Path, required=True,
                        help="exact pre-alignment student checkpoint from the original warm-up recipe")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True,
                        help="selected serving heads export matching the candidate full-state checkpoint")
    parser.add_argument("--candidate-state-checkpoint", type=Path, required=True,
                        help="selected full-state checkpoint named by the best-checkpoint receipt")
    parser.add_argument("--candidate-receipt", type=Path, required=True,
                        help="best-checkpoint.json receipt binding exact full-state and serving export files")
    parser.add_argument("--base-snapshot", type=Path, required=True,
                        help="local immutable LFM base snapshot containing safetensors and exact config/template")
    parser.add_argument("--base", help="optional local pinned HF base snapshot; default uses DEFAULT_BASE/DEFAULT_REVISION")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, help="optional prefix limit for a predeclared probe file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-new-tokens and --limit must be positive")
    report = evaluate(args)
    print(json.dumps({"out": str(args.out), "probes": report["roles"]["selected_candidate"]["probe_count"],
                      "candidate_gate_pass": report["roles"]["selected_candidate"]["gate_pass"]}))


if __name__ == "__main__":
    main()
