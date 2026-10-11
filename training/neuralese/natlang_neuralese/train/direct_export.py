"""Serving export and strict restoration for ``neuralese_direct`` checkpoints.

The training checkpoint remains the resume authority. This smaller artifact
contains only the learned backbone tensors, direct projections, and the clean
port controls needed to reconstruct the model. It deliberately carries no
runtime qualification or older PortHeads certificate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from ..common.jsonio import canonical_json_bytes


ARTIFACT_SCHEMA = "natlang.neuralese-direct-serving/1"
CHECKPOINT_SCHEMA = "natlang.neuralese-direct/1"
LFM2_ADAPTER = "lfm2-hf-dense-layers/1"
MAPLE_BF16_ADAPTER = "maple-bf16-local-snapshot/1"


def _file_inventory(root: Path) -> dict[str, str]:
    """Hash a local HF snapshot without following symlinks outside the snapshot."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"base snapshot is not a directory: {root}")
    values = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            logical_name = path.relative_to(root).as_posix()
            resolved = path.resolve(strict=True)
            # HF Hub snapshots intentionally link files into their sibling
            # ``blobs`` directory. Accept only those cache-internal links.
            allowed_blob_dirs = {root.parents[2] / "blobs", root.parents[1] / "blobs"}
            if not any(blob_dir in resolved.parents for blob_dir in allowed_blob_dirs):
                raise ValueError(f"base snapshot contains an external symlink: {path}")
            if resolved.is_file():
                values[logical_name] = sha256_file_hex(resolved)
        elif path.is_file():
            values[str(path.relative_to(root))] = sha256_file_hex(path)
    if not any(name.endswith(".safetensors") for name in values):
        raise ValueError("base snapshot contains no safetensors model weights")
    if not any(name in values for name in ("tokenizer.json", "tokenizer.model", "spiece.model")):
        raise ValueError("base snapshot has no recognized tokenizer payload")
    return values


def _model_input_inventory(root: Path) -> dict[str, str]:
    """Hash the model/tokenizer inputs that define local HF-style inference.

    Documentation, images and cache metadata are intentionally omitted. Every
    model weight shard, the shard index, architecture config and tokenizer
    inputs are included; an owner proof may bind an even wider inventory.
    """
    root = root.resolve(strict=True)
    names = set()
    for path in root.rglob("*"):
        if not path.is_file() or not (path.suffix == ".safetensors" or path.name in {
                "config.json", "generation_config.json", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer.model", "spiece.model",
                "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                "vocab.json", "merges.txt", "added_tokens.json"}):
            continue
        logical_name = path.relative_to(root).as_posix()
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if root not in resolved.parents:
                raise ValueError(f"Maple base snapshot contains an external symlink: {path}")
        names.add(logical_name)
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("local base snapshot contains no safetensors model weights")
    return {name: sha256_file_hex(root / name) for name in sorted(names)}


def _maple_temporal_binding(proof: dict, checkpoint_sha: str, snapshot: Path,
                            files: dict[str, str]) -> dict:
    """Describe whether today's local Maple base is proven identical to train-time bytes."""
    relation = proof.get("source_temporal_relation")
    if relation == "current-snapshot-reconstructed-training-bytes-unproven":
        return {"relation": relation, "training_base_equivalence_proven": False,
                "exact_training_base_restore": False}
    if relation != "owner-confirmed-same-bytes-used-at-training":
        raise ValueError("Maple source proof must state whether base bytes are historically pinned or reconstructed")
    receipt_path = proof.get("training_base_confirmation_path")
    receipt_sha = proof.get("training_base_confirmation_sha256")
    if not receipt_path or not receipt_sha or sha256_file_hex(receipt_path) != receipt_sha:
        raise ValueError("owner-confirmed Maple base identity requires a pinned training-base confirmation receipt")
    receipt = json.loads(Path(receipt_path).read_text())
    required = {"schema": "natlang.maple-training-base-confirmation/1",
                "checkpoint_sha256": checkpoint_sha, "model_path": str(snapshot),
                "files_sha256": canonical_json_sha256_hex(files),
                "same_bytes_used_during_training": True}
    if any(receipt.get(name) != value for name, value in required.items()):
        raise ValueError("Maple training-base confirmation does not bind this checkpoint and exact model files")
    return {"relation": relation, "training_base_equivalence_proven": True,
            "exact_training_base_restore": True,
            "confirmation_path": str(Path(receipt_path).resolve()),
            "confirmation_sha256": receipt_sha}


def _save_torch_atomic(value, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(value, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_direct_checkpoint(checkpoint_path: str | Path, output_dir: str | Path, *,
                             base_snapshot: str | Path | None, base_source_proof: dict,
                             backbone_adapter: str,
                             selection_receipt: str | Path | None = None) -> dict:
    """Export a direct-stage checkpoint after binding its exact base snapshot.

    ``base_source_proof`` is supplementary provenance supplied by the caller
    (for example a frozen container launch and HF snapshot receipt). Direct
    checkpoints predating base identity fields do not magically acquire those
    fields: the manifest labels the binding as externally reconstructed.
    """
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite direct export: {output_dir}")
    if backbone_adapter not in (LFM2_ADAPTER, MAPLE_BF16_ADAPTER):
        raise NotImplementedError("unsupported direct backbone adapter: " + str(backbone_adapter))
    if not isinstance(base_source_proof, dict) or not base_source_proof:
        raise ValueError("an explicit pinned base source proof is required")
    if base_source_proof.get("backbone_adapter") != backbone_adapter:
        raise ValueError("base source proof adapter does not match the requested backbone adapter")
    checkpoint_sha = sha256_file_hex(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("identity", {}).get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported direct-stage checkpoint schema")
    identity = state.get("identity")
    options = identity.get("options") or {}
    inputs = identity.get("inputs") or {}
    anchor_value = options.get("heads")
    anchor_path = Path(anchor_value).resolve(strict=True) if anchor_value else None
    if anchor_path is None or str(anchor_path) not in inputs:
        raise ValueError("checkpoint identity does not pin its clean-head anchor")
    if sha256_file_hex(anchor_path) != inputs[str(anchor_path)]:
        raise ValueError("clean-head anchor changed since direct training")
    if not isinstance(state.get("student_parameters"), dict) or not state["student_parameters"]:
        raise ValueError("direct checkpoint has no student backbone parameters")
    if not isinstance(state.get("neuralese_projection"), dict):
        raise ValueError("direct checkpoint has no NeuraleseProjection")
    if state.get("token_choice_projection") is not None:
        raise NotImplementedError("the current DirectWriter does not consume TokenChoiceProjection; refusing an unusable serving export")
    package_root = Path(__file__).resolve().parents[1]
    training_parameter_coverage = {"profile": "lfm-full-layer-policy/1",
                                   "attention_trained": True,
                                   "complete_for_declared_policy": True,
                                   "missing_declared_names": []}
    if backbone_adapter == LFM2_ADAPTER:
        required_restore_code = ("model/lfm2_port.py", "train/backbone_policy.py", "serve/__init__.py")
    else:
        required_restore_code = ("maple/student.py", "maple/maple_port.py", "maple/model.py",
                                 "model/hf_port.py", "serve/__init__.py")
    code_pins = identity.get("code") or {}
    for relative in required_restore_code:
        current = sha256_file_hex(package_root / relative)
        if code_pins.get(relative) != current:
            raise ValueError(f"restore-relevant source differs from training checkpoint: {relative}")
    projection_source = package_root / "model/neuralese_projection.py"
    projection_code = {"training_checkpoint_sha256": code_pins.get("model/neuralese_projection.py"),
                       "export_source_sha256": sha256_file_hex(projection_source),
                       "state_shape_checked_on_strict_restore": True}

    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False, mmap=True)
    from ..model.lfm2_port import DEFAULT_BASE
    expected_base = base_source_proof.get("model_id")
    expected_revision = None
    source_binding = None
    if backbone_adapter == LFM2_ADAPTER:
        if base_snapshot is None:
            raise ValueError("LFM export requires an explicit HF base snapshot")
        snapshot = Path(base_snapshot).resolve(strict=True)
        revision = snapshot.name
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("base_snapshot must be an HF snapshot directory named by its immutable revision")
        files = _file_inventory(snapshot)
        expected_revision = base_source_proof.get("revision")
        if expected_base != DEFAULT_BASE or expected_revision != revision:
            raise ValueError("base source proof must bind the declared LFM base model and exact snapshot revision")
        weights_sha = base_source_proof.get("model_safetensors_sha256")
        if weights_sha not in set(files.values()):
            raise ValueError("base source proof model weight digest is absent from the actual snapshot")
        proof_path = base_source_proof.get("launch_receipt_path")
        proof_sha = base_source_proof.get("launch_receipt_sha256")
        if not proof_path or not proof_sha or sha256_file_hex(proof_path) != proof_sha:
            raise ValueError("base source proof must pin a readable launch receipt")
        launch_receipt = json.loads(Path(proof_path).read_text())
        if launch_receipt.get("runtime_image") != base_source_proof.get("runtime_image"):
            raise ValueError("base source proof runtime image differs from the launch receipt")
        if launch_receipt.get("hf_hub_cache") != base_source_proof.get("hf_hub_cache"):
            raise ValueError("base source proof HF cache differs from the launch receipt")
        hub_cache = Path(base_source_proof.get("hf_hub_cache", "")).resolve(strict=True)
        expected_snapshot = hub_cache / "models--LiquidAI--LFM2.5-350M" / "snapshots" / revision
        if Path(base_source_proof.get("snapshot_path", "")).resolve() != snapshot or snapshot != expected_snapshot:
            raise ValueError("base source proof does not name the exact explicit snapshot directory")
        source_inputs = launch_receipt.get("inputs") or {}
        if source_inputs.get("heads", {}).get("sha256") != inputs[str(anchor_path)]:
            raise ValueError("launch receipt does not bind the checkpoint's clean-head anchor")
        source_binding = {"kind": "externally-reconstructed-from-launch-and-snapshot-proof",
                          "backbone_adapter": backbone_adapter, "model_id": expected_base,
                          "revision": revision, "snapshot_path_at_export": str(snapshot),
                          "files": files, "files_sha256": canonical_json_sha256_hex(files),
                          "model_config_sha256": files.get("config.json"),
                          "tokenizer_files": {name: digest for name, digest in files.items()
                                              if "tokenizer" in name or "vocab" in name or name.endswith(".model")
                                              or name == "chat_template.jinja"},
                          "model_safetensors_sha256": weights_sha, "source_proof": base_source_proof}
    else:
        maple = anchor.get("backbone") or {}
        if maple.get("backbone") != "maple" or maple.get("precision") != "bf16" or maple.get("student_state"):
            raise ValueError("Maple BF16 direct export requires a clean BF16 Maple/Mellum heads anchor")
        snapshot = Path(maple.get("base", "")).resolve(strict=True)
        if base_snapshot is not None and Path(base_snapshot).resolve(strict=True) != snapshot:
            raise ValueError("explicit Maple base path differs from the checkpoint's clean-head anchor")
        expected_base = str(snapshot)
        files = _model_input_inventory(snapshot)
        declared_files = base_source_proof.get("files")
        if not isinstance(declared_files, dict) or declared_files != files:
            raise ValueError("Maple source proof must pin the exact local model/tokenizer file inventory")
        if base_source_proof.get("model_path") != str(snapshot):
            raise ValueError("Maple source proof path differs from the clean-head anchor")
        if base_source_proof.get("anchor_sha256") != inputs[str(anchor_path)]:
            raise ValueError("Maple source proof does not bind the checkpoint's clean-head anchor")
        if base_source_proof.get("anchor_backbone_identity") != maple:
            raise ValueError("Maple source proof backbone identity differs from the clean-head anchor")
        init = maple.get("init") or {}
        init_path, init_sha = init.get("receipt"), init.get("receipt_sha256")
        if not init_path or not init_sha or sha256_file_hex(init_path) != init_sha:
            raise ValueError("Maple anchor's initialization receipt is missing or changed")
        schema_proof_path = base_source_proof.get("schema_proof_path")
        schema_proof_sha = base_source_proof.get("schema_proof_sha256")
        if not schema_proof_path or not schema_proof_sha or sha256_file_hex(schema_proof_path) != schema_proof_sha:
            raise ValueError("Maple export requires the exact owner schema-proof file")
        schema_proof = json.loads(Path(schema_proof_path).read_text())
        if (schema_proof.get("checkpoint_sha256") != checkpoint_sha or
                schema_proof.get("step") != state.get("step") or schema_proof.get("identity") != identity):
            raise ValueError("Maple schema proof does not bind this direct checkpoint and identity")
        if schema_proof.get("heads") != {"path": str(anchor_path), "sha256": inputs[str(anchor_path)]}:
            raise ValueError("Maple schema proof does not bind this direct checkpoint's head anchor")
        catalog = {name: [list(value.shape), str(value.dtype)]
                   for name, value in sorted(state["student_parameters"].items())}
        if schema_proof.get("student_parameters_catalog") != catalog:
            raise ValueError("Maple schema proof trainable tensor catalog differs from checkpoint")
        projection_catalog = {
            name: [list(value.shape), str(value.dtype)]
            for name, value in sorted(state["neuralese_projection"].items())
        }
        if schema_proof.get("neuralese_projection_catalog") != projection_catalog:
            raise ValueError("Maple schema proof Neuralese projection catalog differs from checkpoint")
        model_config = json.loads((snapshot / "config.json").read_text())
        layer_count = model_config.get("num_hidden_layers")
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("Maple config must declare a positive num_hidden_layers")
        core_suffixes = ("input_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                         "post_attention_layernorm.weight", "mlp.gate.weight", "mlp.experts.gate_up",
                         "mlp.experts.down")
        attention_suffixes = tuple(f"self_attn.{name}_proj.weight" for name in ("q", "k", "v", "o"))
        expected_core = {f"backbone.model.layers.{layer}.{suffix}"
                         for layer in range(layer_count) for suffix in core_suffixes}
        expected_attention = {f"backbone.model.layers.{layer}.{suffix}"
                              for layer in range(layer_count) for suffix in attention_suffixes}
        saved_names = set(state["student_parameters"])
        if saved_names == expected_core:
            if code_pins.get("train/backbone_policy.py") != \
                    "f27b927250406ae1e6b49aa21656e19882a0c769c225c3b65b094f2d5fec8cd9":
                raise ValueError("the q/k/v/o omission profile is accepted only for the schema-pinned V1 selector")
            training_parameter_coverage = {
                "profile": "maple-bf16-mellum-direct-v1-attention-omitted/1",
                "attention_trained": False, "complete_for_declared_policy": False,
                "missing_declared_names": sorted(expected_attention),
                "omission_pattern": "q-k-v-o-projection-weights-absent-from-checkpoint/1",
                "expected_latent_policy_names": sorted(expected_core | expected_attention)}
        elif saved_names == expected_core | expected_attention:
            training_parameter_coverage = {
                "profile": "maple-bf16-mellum-direct-full-attention/1",
                "attention_trained": True, "complete_for_declared_policy": True,
                "missing_declared_names": [],
                "expected_latent_policy_names": sorted(expected_core | expected_attention)}
        else:
            raise ValueError("Maple direct checkpoint names do not match either explicit supported per-layer tensor profile")
        declared_base_files = schema_proof.get("base_model", {}).get("files") or {}
        if any(files.get(name) != digest for name, digest in declared_base_files.items()):
            raise ValueError("Maple schema proof base model/tokenizer file hashes differ from current snapshot")
        temporal_binding = _maple_temporal_binding(base_source_proof, checkpoint_sha, snapshot, files)
        source_binding = {"kind": "externally-reconstructed-from-closed-owner-schema-proof",
                          "backbone_adapter": backbone_adapter, "model_id": expected_base,
                          "snapshot_path_at_export": str(snapshot), "files": files,
                          "files_sha256": canonical_json_sha256_hex(files),
                          "schema_proof_path": str(Path(schema_proof_path).resolve()),
                          "schema_proof_sha256": schema_proof_sha,
                          "schema_proof_binding": {"checkpoint_sha256": schema_proof["checkpoint_sha256"],
                                                   "step": schema_proof["step"],
                                                   "identity_sha256": canonical_json_sha256_hex(schema_proof["identity"]),
                                                   "heads": schema_proof["heads"],
                                                   "student_parameters_catalog": schema_proof["student_parameters_catalog"],
                                                   "neuralese_projection_catalog": schema_proof["neuralese_projection_catalog"],
                                                   "base_model": schema_proof["base_model"]},
                          "anchor_backbone_identity": maple,
                          "temporal_binding": temporal_binding,
                          "source_proof": base_source_proof,
                          "parameter_selector_correction_commit": "a897248888297c8989230a3534dea5fb3545a52f",
                          "training_parameter_selector_sha256": code_pins.get("train/backbone_policy.py"),
                          "export_parameter_selector_sha256": sha256_file_hex(package_root / "train/backbone_policy.py")}
        quantization_option = options.get("quantization")
        source_binding["inference_profile"] = {
            "backbone_precision": maple["precision"],
            "neuralese_representation": "native-latents",
            "inherits_training_quantization_or_gate": False}
        if quantization_option:
            source_binding["training_precision"] = {
                "mode": "quantization-component-point-distribution-sampled-during-training",
                "identity_option": quantization_option,
                "component_content_sha256": None,
                "inference_artifact_precision_is_separate": True}
            if isinstance(quantization_option, str) and not quantization_option.lstrip().startswith("{"):
                quant_path = Path(quantization_option)
                if not quant_path.is_absolute():
                    quant_path = Path(__file__).resolve().parents[4] / quant_path
                if quant_path.is_file():
                    source_binding["training_precision"]["component_content_sha256"] = sha256_file_hex(quant_path)
                    source_binding["training_precision"]["component_path"] = str(quant_path.resolve())
        else:
            source_binding["training_precision"] = {
                "mode": "BF16-without-quantization-component",
                "identity_option": quantization_option,
                "inference_artifact_precision_is_separate": True}
        training_parameter_coverage["exact_training_base_restore"] = temporal_binding["exact_training_base_restore"]
        training_parameter_coverage["non_trainable_base_values_source"] = (
            "the current locally hashed base snapshot; exact training-time identity is not proven"
            if not temporal_binding["exact_training_base_restore"] else
            "the owner-confirmed exact training-time base snapshot")

    controls = {
        "control_rows": anchor["control_rows"].detach().cpu().contiguous(),
        "control_head_rows": (None if anchor.get("control_head_rows") is None else
                              anchor["control_head_rows"].detach().cpu().contiguous()),
        "port_config": dict(anchor.get("port_config") or {}),
    }
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "student_parameters": {k: v.detach().cpu().contiguous() for k, v in state["student_parameters"].items()},
        "neuralese_projection": {k: v.detach().cpu().contiguous() for k, v in state["neuralese_projection"].items()},
        "token_choice_projection": (None if state.get("token_choice_projection") is None else
                                     {k: v.detach().cpu().contiguous()
                                      for k, v in state["token_choice_projection"].items()}),
        **controls,
    }
    tensor_catalog = {}
    for group in ("student_parameters", "neuralese_projection", "token_choice_projection"):
        values = payload[group]
        if values is None:
            tensor_catalog[group] = None
        else:
            tensor_catalog[group] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                     for name, value in sorted(values.items())}
    tensor_catalog["control_rows"] = {"shape": list(payload["control_rows"].shape),
                                      "dtype": str(payload["control_rows"].dtype)}
    tensor_catalog["control_head_rows"] = (None if payload["control_head_rows"] is None else
                                            {"shape": list(payload["control_head_rows"].shape),
                                             "dtype": str(payload["control_head_rows"].dtype)})

    selection = None
    if selection_receipt:
        selection_path = Path(selection_receipt).resolve(strict=True)
        selection = {"path": str(selection_path), "sha256": sha256_file_hex(selection_path),
                     "document": json.loads(selection_path.read_text())}
        if selection["document"].get("step") != state.get("step"):
            raise ValueError("selected checkpoint receipt step differs from exported checkpoint")
        selected_file = selection["document"].get("file")
        if selected_file and Path(selected_file).name != checkpoint_path.name:
            raise ValueError("selected checkpoint receipt names a different file")
    output_dir.mkdir(parents=True, exist_ok=False)
    payload_path = output_dir / "direct-tensors.pt"
    _save_torch_atomic(payload, payload_path)
    payload_sha = sha256_file_hex(payload_path)

    projection_config = {"blocks": int(options.get("projection_blocks", 2)),
                         "hidden": int(options.get("projection_hidden", 0)) or None,
                         "kernel": int(options.get("projection_kernel", 4))}
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "status": "direct-writer-pending",
        "training_checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha,
                                "schema": state["schema"], "step": int(state["step"]),
                                "identity_sha256": canonical_json_sha256_hex(identity),
                                "base_identity_in_checkpoint": False},
        "selection_receipt": selection,
        "base_binding": source_binding,
        "anchor": {"path": str(anchor_path), "sha256": inputs[str(anchor_path)]},
        "identity": identity,
        "projection": {"neuralese": projection_config,
                       "token_choice": (None if payload["token_choice_projection"] is None else
                                        {**projection_config, "present": True})},
        "projection_implementation": projection_code,
        "port_config": payload["port_config"],
        "tensor_catalog": tensor_catalog,
        "training_parameter_coverage": training_parameter_coverage,
        "payload": {"path": payload_path.name, "sha256": payload_sha},
        "qualification": {"runtime": False, "data_admission": False,
                           "inherits_legacy_portheads_certificate": False},
    }
    manifest["manifest_sha256"] = canonical_json_sha256_hex(manifest)
    _write_json_atomic(manifest, output_dir / "manifest.json")
    return manifest


@dataclass
class DirectServingModel:
    """Restored tensors plus the ordinary LFM engine; serving writer remains pending."""
    engine: object
    neuralese_projection: torch.nn.Module
    token_choice_projection: torch.nn.Module | None
    manifest: dict
    status: str = "direct-writer-pending"


def load_direct_artifact(artifact_dir: str | Path, *, base_snapshot: str | Path,
                         device: str = "cpu", dtype=None) -> DirectServingModel:
    """Strictly restore a direct export into the pinned clean-base engine."""
    artifact_dir = Path(artifact_dir).resolve(strict=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest.pop("manifest_sha256", None)
    if manifest.get("schema") != ARTIFACT_SCHEMA or claimed != canonical_json_sha256_hex(manifest):
        raise ValueError("direct artifact manifest schema/hash mismatch")
    manifest["manifest_sha256"] = claimed
    if manifest.get("status") != "direct-writer-pending":
        raise ValueError("direct artifact status is not supported by this loader")
    base = manifest["base_binding"]
    adapter = base.get("backbone_adapter")
    from ..model.lfm2_port import DEFAULT_BASE
    # Only the first local artifact predates the explicit adapter discriminator.
    # Its exact model ID and revision make this LFM restore branch unambiguous.
    if adapter is None and base.get("model_id") == DEFAULT_BASE:
        adapter = LFM2_ADAPTER
    if adapter not in (LFM2_ADAPTER, MAPLE_BF16_ADAPTER):
        raise NotImplementedError("this artifact's backbone adapter has no implemented restore path")
    snapshot = Path(base_snapshot).resolve(strict=True)
    proof = base.get("source_proof") or {}
    if adapter == LFM2_ADAPTER:
        proof_path, proof_sha = proof.get("launch_receipt_path"), proof.get("launch_receipt_sha256")
        if (proof.get("backbone_adapter", LFM2_ADAPTER) != LFM2_ADAPTER or
                proof.get("model_id") != DEFAULT_BASE or proof.get("revision") != snapshot.name or
                Path(proof.get("snapshot_path", "")).resolve() != snapshot or
                not proof_path or not proof_sha or sha256_file_hex(proof_path) != proof_sha):
            raise ValueError("direct artifact LFM source proof is missing or changed")
        launch_receipt = json.loads(Path(proof_path).read_text())
        if (launch_receipt.get("runtime_image") != proof.get("runtime_image") or
                launch_receipt.get("hf_hub_cache") != proof.get("hf_hub_cache")):
            raise ValueError("direct artifact launch receipt differs from its LFM source proof")
        cache = Path(proof["hf_hub_cache"]).resolve(strict=True)
        expected_snapshot = cache / "models--LiquidAI--LFM2.5-350M" / "snapshots" / snapshot.name
        if snapshot != expected_snapshot or snapshot.name != base["revision"] or _file_inventory(snapshot) != base["files"]:
            raise ValueError("loaded base snapshot differs from direct artifact's exact file inventory")
    else:
        if str(snapshot) != base.get("model_id") or str(snapshot) != base.get("snapshot_path_at_export"):
            raise ValueError("loaded Maple base path differs from direct artifact source proof")
        loaded_files = _model_input_inventory(snapshot)
        if loaded_files != base.get("files"):
            raise ValueError("loaded Maple model/tokenizer files differ from direct artifact's exact file inventory")
        schema_proof = base.get("schema_proof_binding") or {}
        if schema_proof.get("checkpoint_sha256") != manifest["training_checkpoint"]["sha256"]:
            raise ValueError("embedded Maple schema proof does not bind the exported training checkpoint")
        if (schema_proof.get("identity_sha256") != canonical_json_sha256_hex(manifest.get("identity")) or
                schema_proof.get("step") != manifest["training_checkpoint"].get("step")):
            raise ValueError("embedded Maple schema proof identity or step differs from the artifact")
        temporal_binding = _maple_temporal_binding(proof, manifest["training_checkpoint"]["sha256"],
                                                   snapshot, loaded_files)
        if temporal_binding != base.get("temporal_binding"):
            raise ValueError("Maple training-base temporal proof differs from the exported binding")
        inference_profile = base.get("inference_profile") or {}
        if (inference_profile.get("backbone_precision") != "bf16" or
                inference_profile.get("neuralese_representation") != "native-latents" or
                inference_profile.get("inherits_training_quantization_or_gate") is not False):
            raise ValueError("Maple direct artifact inference precision profile is missing or inconsistent")
        training_precision = base.get("training_precision") or {}
        quantization_option = (manifest.get("identity", {}).get("options") or {}).get("quantization")
        expected_training_precision_mode = ("quantization-component-point-distribution-sampled-during-training"
                                            if quantization_option else "BF16-without-quantization-component")
        if (training_precision.get("identity_option") != quantization_option or
                training_precision.get("mode") != expected_training_precision_mode or
                training_precision.get("inference_artifact_precision_is_separate") is not True):
            raise ValueError("Maple inference profile conflates training quantization with serving precision")
        component_sha = training_precision.get("component_content_sha256")
        component_path = training_precision.get("component_path")
        if component_sha is not None and (not component_path or sha256_file_hex(component_path) != component_sha):
            raise ValueError("Maple training quantization component changed after export")
        proof_heads = schema_proof.get("heads") or {}
        if proof_heads.get("sha256") != manifest["anchor"]["sha256"]:
            raise ValueError("embedded Maple schema proof does not bind the clean-head anchor")
        proof_catalog = schema_proof.get("student_parameters_catalog") or {}
        actual_catalog = {name: [item["shape"], item["dtype"]]
                          for name, item in manifest["tensor_catalog"]["student_parameters"].items()}
        if proof_catalog != actual_catalog:
            raise ValueError("embedded Maple schema proof tensor catalog differs from exported payload")
        proof_projection_catalog = schema_proof.get("neuralese_projection_catalog") or {}
        actual_projection_catalog = {
            name: [item["shape"], item["dtype"]]
            for name, item in manifest["tensor_catalog"]["neuralese_projection"].items()
        }
        if proof_projection_catalog != actual_projection_catalog:
            raise ValueError("embedded Maple schema proof Neuralese projection catalog differs from exported payload")
        coverage = manifest.get("training_parameter_coverage") or {}
        missing_declared = coverage.get("missing_declared_names")
        if missing_declared:
            if (coverage.get("profile") != "maple-bf16-mellum-direct-v1-attention-omitted/1" or
                    coverage.get("attention_trained") is not False or
                    coverage.get("complete_for_declared_policy") is not False or
                    coverage.get("omission_pattern") != "q-k-v-o-projection-weights-remain-at-pinned-base-values/1"):
                raise ValueError("Maple partial parameter coverage lacks its exact reviewed omission profile")
        elif (coverage.get("profile") != "maple-bf16-mellum-direct-full-attention/1" or
              coverage.get("attention_trained") is not True or
              coverage.get("complete_for_declared_policy") is not True):
            raise ValueError("Maple complete parameter coverage declaration is inconsistent")
    payload_path = artifact_dir / manifest["payload"]["path"]
    if sha256_file_hex(payload_path) != manifest["payload"]["sha256"]:
        raise ValueError("direct artifact tensor payload hash mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("direct tensor payload schema mismatch")

    from ..serve import load_engine
    from ..model.neuralese_projection import NeuraleseProjection
    from .backbone_policy import full_backbone_parameter_names, restore_backbone_trainables

    anchor = manifest["anchor"]
    if sha256_file_hex(anchor["path"]) != anchor["sha256"]:
        raise ValueError("clean-head anchor changed after direct export")
    if adapter == MAPLE_BF16_ADAPTER:
        anchor_state = torch.load(anchor["path"], map_location="cpu", weights_only=False, mmap=True)
        if anchor_state.get("backbone") != base.get("anchor_backbone_identity"):
            raise ValueError("Maple base identity differs from its pinned clean-head anchor")
        init = anchor_state["backbone"].get("init") or {}
        if not init.get("receipt") or sha256_file_hex(init["receipt"]) != init.get("receipt_sha256"):
            raise ValueError("Maple anchor initialization receipt is missing or changed")
        del anchor_state
    engine = load_engine(base=str(snapshot), heads_checkpoint=anchor["path"], device=device, dtype=dtype)
    if engine.heads.read_markers or getattr(engine.heads, "read_adapter", None) is not None:
        raise ValueError("direct-v1 artifact currently requires raw-token heads without a read adapter")
    saved_parameters = payload["student_parameters"]
    prefix = "backbone."
    if any(not name.startswith(prefix) for name in saved_parameters):
        raise ValueError("direct checkpoint trainable names must retain their declared backbone. prefix")
    native_parameters = {name[len(prefix):]: value for name, value in saved_parameters.items()}
    if len(native_parameters) != len(saved_parameters):
        raise ValueError("direct checkpoint trainable names collide after backbone prefix removal")
    if adapter == LFM2_ADAPTER:
        expected = full_backbone_parameter_names(engine.backbone)
        if set(native_parameters) != expected:
            raise ValueError("LFM direct checkpoint trainable names differ from the declared full-layer policy")
    else:
        from .backbone_policy import latent_backbone_parameters
        declared = {name for name, _ in latent_backbone_parameters(engine.backbone)}
        # Restore into the original BF16 Mellum. A run-v1 trainer ordering bug
        # omitted only q/k/v/o attention projections; their values remain the
        # unchanged source-base values. Later direct runs may save the complete
        # latent policy. Any other partial/mismatched name set is rejected.
        allowed_attention = {name for name in declared if any(
            name.endswith(f".self_attn.{projection}_proj.weight")
            for projection in ("q", "k", "v", "o"))}
        coverage = manifest.get("training_parameter_coverage") or {}
        missing = declared - set(native_parameters)
        if sorted(declared) != coverage.get("expected_latent_policy_names"):
            raise ValueError("Maple latent policy parameter names differ from the export coverage declaration")
        if sorted(missing) != coverage.get("missing_declared_names"):
            raise ValueError("Maple omitted trainables differ from the export coverage declaration")
        if set(native_parameters) not in (declared, declared - allowed_attention):
            raise ValueError("Maple direct trainable names must match the complete latent policy or its declared q/k/v/o omission")
        if bool(coverage.get("complete_for_declared_policy")) != (not missing) or \
                bool(coverage.get("attention_trained")) != (not (missing & allowed_attention)):
            raise ValueError("Maple training coverage flags conflict with the exact saved parameter set")
        if missing:
            if (missing != allowed_attention or
                    coverage.get("profile") != "maple-bf16-mellum-direct-v1-attention-omitted/1" or
                    coverage.get("omission_pattern") != "q-k-v-o-projection-weights-absent-from-checkpoint/1"):
                raise ValueError("Maple partial restore is accepted only for the schema-bound q/k/v/o omission profile")
        elif coverage.get("profile") != "maple-bf16-mellum-direct-full-attention/1":
            raise ValueError("Maple complete restore has an inconsistent trainable-coverage profile")
        if coverage.get("exact_training_base_restore") != temporal_binding["exact_training_base_restore"]:
            raise ValueError("Maple checkpoint base-fidelity flag conflicts with the source temporal proof")
        from .backbone_policy import __file__ as selector_source
        if sha256_file_hex(selector_source) != base.get("export_parameter_selector_sha256"):
            raise ValueError("Maple restore parameter selector changed after export")
        for parameter in engine.backbone.parameters():
            parameter.requires_grad_(False)
        expected = set(native_parameters)
    restore_backbone_trainables(engine.backbone, native_parameters, expected_names=expected)
    width = int(engine.backbone.hf.config.hidden_size)
    neuralese = NeuraleseProjection(width, **manifest["projection"]["neuralese"])
    neuralese.load_state_dict(payload["neuralese_projection"], strict=True)
    neuralese.to(device=device).eval()
    for parameter in neuralese.parameters():
        parameter.requires_grad_(False)
    token_choice = None
    if manifest["projection"]["token_choice"] is not None:
        raise NotImplementedError("TokenChoiceProjection artifacts cannot be loaded by the current DirectWriter")
    if not torch.equal(engine.backbone.control_rows.cpu(), payload["control_rows"].cpu()):
        raise ValueError("loaded clean-base controls differ from exported control rows")
    expected_head_controls = payload["control_head_rows"]
    actual_head_controls = getattr(engine.backbone, "control_head_rows", None)
    if (expected_head_controls is None) != (actual_head_controls is None):
        raise ValueError("loaded clean-base control-head layout differs from export")
    if expected_head_controls is not None and not torch.equal(actual_head_controls.cpu(),
                                                             expected_head_controls.cpu()):
        raise ValueError("loaded clean-base control-head rows differ from export")
    actual_port_config = engine.heads.port_config()
    actual_port_config.update({"cutoff": engine.heads.cutoff,
                               "max_length": engine.heads.max_length})
    if actual_port_config != manifest["port_config"]:
        raise ValueError("restored PortHeads control metadata differs from direct export")
    # Shared Engine dispatch can opt into the direct writer only after that
    # writer has been independently reviewed; until then the manifest stays
    # explicitly pending and ordinary HF generation remains a separate path.
    engine.neuralese_projection = neuralese
    engine.token_choice_projection = token_choice
    engine.direct_artifact_manifest = manifest
    return DirectServingModel(engine, neuralese, token_choice, manifest)
