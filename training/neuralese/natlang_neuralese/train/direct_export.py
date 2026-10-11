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
                             base_snapshot: str | Path, base_source_proof: dict,
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
    if backbone_adapter != LFM2_ADAPTER:
        raise NotImplementedError("direct export currently implements only the pinned LFM2 dense-layer adapter; "
                                  "Maple/Mellum base identity and trainable restore need their own adapter")
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
    required_restore_code = ("model/lfm2_port.py", "train/backbone_policy.py", "serve/__init__.py")
    code_pins = identity.get("code") or {}
    for relative in required_restore_code:
        current = sha256_file_hex(package_root / relative)
        if code_pins.get(relative) != current:
            raise ValueError(f"restore-relevant source differs from training checkpoint: {relative}")
    projection_source = package_root / "model/neuralese_projection.py"
    projection_code = {"training_checkpoint_sha256": code_pins.get("model/neuralese_projection.py"),
                       "export_source_sha256": sha256_file_hex(projection_source),
                       "state_shape_checked_on_strict_restore": True}

    snapshot = Path(base_snapshot).resolve(strict=True)
    revision = snapshot.name
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("base_snapshot must be an HF snapshot directory named by its immutable revision")
    files = _file_inventory(snapshot)
    from ..model.lfm2_port import DEFAULT_BASE
    expected_base = base_source_proof.get("model_id")
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
    if weights_sha not in set(files.values()):
        raise ValueError("base model weight digest is not present in the frozen HF snapshot")

    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False, mmap=True)
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
        "base_binding": {"kind": "externally-reconstructed-from-launch-and-snapshot-proof",
                         "backbone_adapter": backbone_adapter,
                         "model_id": expected_base, "revision": revision,
                         "snapshot_path_at_export": str(snapshot), "files": files,
                         "files_sha256": canonical_json_sha256_hex(files),
                         "model_config_sha256": files.get("config.json"),
                         "tokenizer_files": {name: digest for name, digest in files.items()
                                             if "tokenizer" in name or "vocab" in name or name.endswith(".model")
                                             or name == "chat_template.jinja"},
                         "model_safetensors_sha256": weights_sha,
                         "source_proof": base_source_proof},
        "anchor": {"path": str(anchor_path), "sha256": inputs[str(anchor_path)]},
        "identity": identity,
        "projection": {"neuralese": projection_config,
                       "token_choice": (None if payload["token_choice_projection"] is None else
                                        {**projection_config, "present": True})},
        "projection_implementation": projection_code,
        "port_config": payload["port_config"],
        "tensor_catalog": tensor_catalog,
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
    if adapter != LFM2_ADAPTER:
        raise NotImplementedError("this artifact's backbone adapter has no implemented restore path")
    snapshot = Path(base_snapshot).resolve(strict=True)
    proof = base.get("source_proof") or {}
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
    engine = load_engine(base=str(snapshot), heads_checkpoint=anchor["path"], device=device, dtype=dtype)
    if engine.heads.read_markers or getattr(engine.heads, "read_adapter", None) is not None:
        raise ValueError("direct-v1 artifact currently requires raw-token heads without a read adapter")
    expected = full_backbone_parameter_names(engine.backbone)
    saved_parameters = payload["student_parameters"]
    prefix = "backbone."
    if any(not name.startswith(prefix) for name in saved_parameters):
        raise ValueError("direct checkpoint trainable names must retain their declared backbone. prefix")
    native_parameters = {name[len(prefix):]: value for name, value in saved_parameters.items()}
    if len(native_parameters) != len(saved_parameters):
        raise ValueError("direct checkpoint trainable names collide after backbone prefix removal")
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
