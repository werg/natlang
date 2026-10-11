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
RECURRENCE_CHECKPOINT_SCHEMA = "natlang.neuralese_recurrence_checkpoint/1"
DIRECT_PARENT_BINDING_SCHEMA = "natlang.neuralese-direct-parent-binding/1"
RECURRENCE_PARENT_PROOF_SCHEMA = "natlang.neuralese-direct-recurrence-parent-proof/1"
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


def direct_parent_binding(artifact_dir: str | Path) -> dict:
    """Return a content-bound document for the exact direct artifact loaded by a recurrence run."""
    artifact_dir = Path(artifact_dir).resolve(strict=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest_bytes_sha = sha256_file_hex(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest.pop("manifest_sha256", None)
    if manifest.get("schema") != ARTIFACT_SCHEMA or claimed != canonical_json_sha256_hex(manifest):
        raise ValueError("direct parent manifest schema/hash mismatch")
    manifest["manifest_sha256"] = claimed
    if manifest.get("status") != "direct-writer-pending":
        raise ValueError("direct parent manifest has an unsupported serving status")
    payload = manifest.get("payload") or {}
    payload_name = Path(str(payload.get("path", "")))
    if not payload.get("path") or payload_name.is_absolute() or ".." in payload_name.parts:
        raise ValueError("direct parent payload path must stay inside its artifact directory")
    payload_path = artifact_dir / payload_name
    if not payload_path.is_file() or sha256_file_hex(payload_path) != payload.get("sha256"):
        raise ValueError("direct parent tensor payload is missing or changed")
    return {
        "schema": DIRECT_PARENT_BINDING_SCHEMA,
        "artifact_path": str(artifact_dir),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": manifest_bytes_sha,
        "manifest_sha256": claimed,
        "payload_path": str(payload_path.resolve()),
        "payload_sha256": payload["sha256"],
        "manifest": manifest,
    }


def verify_direct_parent_binding(binding: dict, current: dict | None = None) -> dict:
    """Validate an embedded parent binding and, when supplied, the current declared parent."""
    if not isinstance(binding, dict) or binding.get("schema") != DIRECT_PARENT_BINDING_SCHEMA:
        raise ValueError("recurrence checkpoint has no supported direct-parent binding")
    manifest_path_text = binding.get("manifest_path")
    if not isinstance(manifest_path_text, str) or not manifest_path_text:
        raise ValueError("embedded direct-parent binding has no provenance manifest path")
    manifest_path = Path(manifest_path_text)
    saved_manifest = binding.get("manifest")
    if not isinstance(saved_manifest, dict):
        raise ValueError("embedded direct-parent binding has no manifest document")
    unsigned = dict(saved_manifest)
    saved_manifest_sha = unsigned.pop("manifest_sha256", None)
    if (saved_manifest.get("schema") != ARTIFACT_SCHEMA or
            saved_manifest_sha != binding.get("manifest_sha256") or
            saved_manifest_sha != canonical_json_sha256_hex(unsigned)):
        raise ValueError("embedded direct-parent manifest schema/hash mismatch")
    payload = saved_manifest.get("payload") or {}
    if payload.get("sha256") != binding.get("payload_sha256"):
        raise ValueError("embedded direct-parent payload hash differs from its manifest")
    # Paths are retained as historical provenance; the explicitly declared
    # current artifact resolves relocation. Parent identity is content-based.
    if current is None:
        actual = direct_parent_binding(manifest_path.parent.resolve(strict=True))
    else:
        actual = current
    for key in ("manifest_file_sha256", "manifest_sha256", "payload_sha256", "manifest"):
        if binding.get(key) != actual.get(key):
            raise ValueError("embedded direct-parent content changed: " + key)
    return actual


def verify_recurrence_parent_proof(proof: dict, *, checkpoint_path: str | Path,
                                   checkpoint_sha256: str, current_parent: dict,
                                   checkpoint_identity: dict | None = None) -> dict:
    """Verify a declared stage/launch chain for a legacy recurrence checkpoint.

    This is a migration path for checkpoints created before they embedded their
    direct parent. The packet must bind the checkpoint bytes, parent manifest,
    immutable stage plan, launch intent, executed launch receipt, and recipe.
    """
    if not isinstance(proof, dict) or proof.get("schema") != RECURRENCE_PARENT_PROOF_SCHEMA:
        raise ValueError("legacy recurrence export requires a declared parent-proof packet")
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    checkpoint_ref = proof.get("checkpoint")
    if (not isinstance(checkpoint_ref, dict) or
            not isinstance(checkpoint_ref.get("path"), str) or
            checkpoint_ref.get("sha256") != checkpoint_sha256):
        raise ValueError("parent-proof packet does not bind these recurrence checkpoint bytes")

    def bound_file(name: str) -> tuple[Path, dict]:
        row = proof.get(name)
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise ValueError("parent-proof packet lacks a pinned " + name)
        path = Path(row["path"]).resolve(strict=True)
        if sha256_file_hex(path) != row["sha256"]:
            raise ValueError("parent-proof packet file changed: " + name)
        return path, row

    plan_path, plan_ref = bound_file("stage_plan")
    intent_path, intent_ref = bound_file("launch_intent")
    launch_path, launch_ref = bound_file("launch_receipt")
    recipe_path, recipe_ref = bound_file("recipe")
    parent_ref = proof.get("parent_manifest")
    if (not isinstance(parent_ref, dict) or not isinstance(parent_ref.get("path"), str) or
            not isinstance(parent_ref.get("sha256"), str)):
        raise ValueError("parent-proof packet lacks a pinned parent manifest")
    parent_path = Path(parent_ref["path"])
    if (parent_ref["sha256"] != current_parent["manifest_file_sha256"] or
            sha256_file_hex(current_parent["manifest_path"]) != parent_ref["sha256"]):
        raise ValueError("parent-proof packet parent-manifest hash differs from the loaded artifact")

    plan = json.loads(plan_path.read_text())
    intent = json.loads(intent_path.read_text())
    launch = json.loads(launch_path.read_text())
    if plan.get("schema") != "natlang.neuralese-direct-stage-plan/1":
        raise ValueError("parent-proof stage plan has an unsupported schema")
    if intent.get("schema") != "natlang.neuralese-direct-stage-launch-intent/1":
        raise ValueError("parent-proof launch intent has an unsupported schema")
    if (os.path.normpath(str(intent.get("plan_path", ""))) != os.path.normpath(str(plan_path)) or
            intent.get("plan_sha256") != plan_ref["sha256"]):
        raise ValueError("launch intent does not bind the exact stage plan")
    stage_inputs = plan.get("inputs") or {}
    declared_parent = stage_inputs.get("direct_artifact_manifest") or {}
    if (os.path.normpath(str(declared_parent.get("path", ""))) != os.path.normpath(str(parent_path)) or
            declared_parent.get("sha256") != parent_ref["sha256"]):
        raise ValueError("stage plan does not bind the exact direct-parent manifest")
    if (intent.get("input_sha256") or {}).get("direct_artifact_manifest") != parent_ref["sha256"]:
        raise ValueError("launch intent does not bind the direct-parent manifest hash")
    recipe_sha = recipe_ref["sha256"]
    if (os.path.normpath(str(plan.get("recipe_path", ""))) != os.path.normpath(str(recipe_path)) or
            plan.get("recipe_sha256") != recipe_sha or intent.get("recipe_sha256") != recipe_sha):
        raise ValueError("stage plan or launch intent does not bind the exact recipe")
    if launch.get("runtime_code") != plan.get("frozen_code"):
        raise ValueError("executed launch receipt code differs from the stage plan")
    if (len(plan.get("frozen_code") or {}) != intent.get("frozen_code_count") or
            canonical_json_sha256_hex(plan.get("frozen_code") or {}) != intent.get("frozen_code_manifest_sha256")):
        raise ValueError("launch intent frozen-code receipt differs from the stage plan")
    argv = launch.get("argv") or []
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("executed launch receipt has no exact argv")
    if ("--recipe" not in argv or str(recipe_path) not in argv or
            "--launch-metadata" not in argv or str(intent_path) not in argv):
        raise ValueError("executed launch argv does not bind the declared recipe and launch intent")
    if (intent.get("runtime_image") != plan.get("runtime_image") or
            intent.get("runtime_image") not in argv or
            "--out" not in argv or os.path.normpath(str(intent.get("output_path", ""))) not in
            [os.path.normpath(arg) for arg in argv]):
        raise ValueError("executed launch argv does not bind the declared runtime image and output root")
    try:
        Path(os.path.normpath(str(plan.get("stage_output", "")))).relative_to(
            Path(os.path.normpath(str(intent.get("output_path", "")))))
        Path(os.path.normpath(str(checkpoint_ref["path"]))).relative_to(
            Path(os.path.normpath(str(plan.get("stage_output", "")))))
    except ValueError as exc:
        raise ValueError("stage output is outside the launch intent output root") from exc
    if checkpoint_identity is not None:
        identity_options = checkpoint_identity.get("options") or {}
        declared_parent_path = identity_options.get("direct_artifact")
        if (not declared_parent_path or
                os.path.normpath(str(Path(declared_parent_path) / "manifest.json")) !=
                os.path.normpath(str(parent_path))):
            raise ValueError("checkpoint identity does not name the direct artifact proven by its stage plan")
        checkpoint_code = _recurrence_code_pins(checkpoint_identity)
        launch_code = launch.get("runtime_code") or {}
        if any(launch_code.get(name) != digest for name, digest in checkpoint_code.items()):
            raise ValueError("checkpoint source-code pins differ from the executed frozen runtime")
    return {
        "schema": RECURRENCE_PARENT_PROOF_SCHEMA,
        "checkpoint": proof["checkpoint"],
        "stage_plan": {"path": str(plan_path), "sha256": plan_ref["sha256"]},
        "launch_intent": {"path": str(intent_path), "sha256": intent_ref["sha256"]},
        "launch_receipt": {"path": str(launch_path), "sha256": launch_ref["sha256"]},
        "recipe": {"path": str(recipe_path), "sha256": recipe_sha},
        "parent_manifest": {"path": str(parent_path), "sha256": parent_ref["sha256"]},
    }


def _recurrence_code_pins(identity: dict) -> dict:
    """Normalize recurrence's absolute package source pins to serving package paths."""
    raw = identity.get("code")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("recurrence identity has no frozen source-code pins")
    output = {}
    for source_path, digest in raw.items():
        if not isinstance(source_path, str) or not isinstance(digest, str):
            raise ValueError("recurrence source-code pins must map paths to SHA-256 strings")
        parts = Path(source_path).parts
        try:
            package_index = parts.index("natlang_neuralese")
        except ValueError as exc:
            raise ValueError("recurrence code pin is outside the natlang_neuralese package") from exc
        relative = Path(*parts[package_index + 1:]).as_posix()
        if not relative or relative in output:
            raise ValueError("recurrence source-code pins contain an empty or duplicate package path")
        output[relative] = digest
    return output


def _recurrence_export_state(state: dict, checkpoint_path: Path, checkpoint_sha: str,
                             recurrence_parent_proof: dict | None,
                             recurrence_parent_artifact: str | Path | None) -> tuple[dict, dict, dict]:
    """Adapt the one declared native direct-recurrence schema into the existing exporter fields."""
    identity = state.get("identity")
    if state.get("schema") != RECURRENCE_CHECKPOINT_SCHEMA or not isinstance(identity, dict):
        raise ValueError("unsupported recurrence checkpoint schema")
    if identity.get("schema") not in (None, RECURRENCE_CHECKPOINT_SCHEMA):
        raise ValueError("recurrence checkpoint identity has an unexpected schema")
    source_options = identity.get("options") or {}
    saved_binding = state.get("direct_parent_binding")
    parent_path = recurrence_parent_artifact or source_options.get("direct_artifact") or (
        saved_binding.get("artifact_path") if isinstance(saved_binding, dict) else None)
    if not isinstance(parent_path, (str, Path)) or not parent_path:
        raise ValueError("recurrence checkpoint was not trained from a declared direct artifact")
    parent_artifact = Path(parent_path).resolve(strict=True)
    parent = direct_parent_binding(parent_artifact)
    migration_proof = None
    if saved_binding is not None:
        verify_direct_parent_binding(saved_binding, parent)
    else:
        if recurrence_parent_proof is None:
            raise ValueError("legacy recurrence checkpoint lacks an embedded parent; supply its pinned stage/launch proof")
        migration_proof = verify_recurrence_parent_proof(
            recurrence_parent_proof, checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha, current_parent=parent,
            checkpoint_identity=identity)

    parent_manifest = parent["manifest"]
    base_binding = parent_manifest.get("base_binding") or {}
    if base_binding.get("backbone_adapter") != LFM2_ADAPTER:
        raise NotImplementedError("recurrence export currently supports only the declared LFM2 serving adapter")
    inherited_source_proof = base_binding.get("source_proof")
    anchor = parent_manifest.get("anchor") or {}
    anchor_path = Path(anchor.get("path", "")).resolve(strict=True)
    anchor_sha = anchor.get("sha256")
    if not anchor_sha or sha256_file_hex(anchor_path) != anchor_sha:
        raise ValueError("parent direct artifact clean-head anchor is missing or changed")
    if (not isinstance(inherited_source_proof, dict) or
            base_binding.get("model_id") != inherited_source_proof.get("model_id") or
            base_binding.get("revision") != inherited_source_proof.get("revision")):
        raise ValueError("parent direct artifact has no consistent inherited base source proof")

    backbone_training = state.get("backbone_training")
    if backbone_training not in ("full", "latent"):
        raise ValueError("recurrence serving export supports only full or latent dense LFM trainables")
    trainables = state.get("backbone_trainables")
    neuralese_projection = state.get("neuralese_projection")
    token_choice_projection = state.get("token_choice_projection")
    if not isinstance(trainables, dict) or not trainables:
        raise ValueError("recurrence checkpoint has no full backbone_trainables state")
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
           for name, value in trainables.items()):
        raise ValueError("recurrence backbone trainables contain invalid names or values")
    student_parameters = {
        (name if name.startswith("backbone.") else "backbone." + name): value
        for name, value in trainables.items()
    }
    if len(student_parameters) != len(trainables):
        raise ValueError("recurrence backbone trainable names collide after applying the backbone. namespace")
    parent_catalog = parent_manifest.get("tensor_catalog") or {}
    expected_backbone = parent_catalog.get("student_parameters")
    if not isinstance(expected_backbone, dict) or not expected_backbone:
        raise ValueError("parent direct artifact has no complete backbone tensor catalog")
    expected_names = set()
    expected_shapes = {}
    for name, entry in expected_backbone.items():
        if not name.startswith("backbone.") or not isinstance(entry, dict) or not isinstance(entry.get("shape"), list):
            raise ValueError("parent direct artifact backbone catalog is malformed")
        relative = name[len("backbone."):]
        if relative in expected_shapes:
            raise ValueError("parent direct artifact backbone catalog has duplicate native names")
        expected_names.add(name)
        expected_shapes[relative] = entry["shape"]
    actual_shapes = {name: list(value.shape) for name, value in trainables.items()}
    if set(student_parameters) != expected_names or actual_shapes != expected_shapes:
        raise ValueError("recurrence backbone trainables differ from the parent's exact complete name/shape catalog")
    if not isinstance(neuralese_projection, dict) or not neuralese_projection:
        raise ValueError("recurrence checkpoint has no NeuraleseProjection state")
    if not isinstance(token_choice_projection, dict) or not token_choice_projection:
        raise ValueError("native direct recurrence checkpoint has no TokenChoiceProjection state")
    projection_catalogs = parent_catalog.get("neuralese_projection")
    if not isinstance(projection_catalogs, dict) or not projection_catalogs:
        raise ValueError("parent direct artifact has no NeuraleseProjection tensor catalog")
    for group_name, values in (("neuralese_projection", neuralese_projection),
                               ("token_choice_projection", token_choice_projection)):
        expected_group = parent_catalog.get(group_name)
        if group_name == "token_choice_projection" and expected_group is None:
            # The current recurrence initializer creates TokenChoiceProjection
            # with the parent's NeuraleseProjection stack config. Both modules
            # share the same declared stack state layout.
            expected_group = projection_catalogs
        if not isinstance(expected_group, dict) or not expected_group:
            raise ValueError("parent or declared initializer has no " + group_name + " tensor catalog")
        actual_group = {name: list(value.shape) for name, value in values.items()
                        if isinstance(value, torch.Tensor)}
        expected_group_shapes = {name: entry.get("shape") for name, entry in expected_group.items()
                                 if isinstance(entry, dict)}
        if len(actual_group) != len(values) or actual_group != expected_group_shapes:
            raise ValueError("recurrence " + group_name + " differs from the parent's exact name/shape catalog")
    control_rows = state.get("control_rows")
    if not isinstance(control_rows, torch.Tensor):
        raise ValueError("recurrence checkpoint has no trained control_rows tensor")
    control_head_rows = state.get("control_head_rows")
    if control_head_rows is not None and not isinstance(control_head_rows, torch.Tensor):
        raise ValueError("recurrence checkpoint control_head_rows is not a tensor")
    port_config = state.get("port_config")
    if not isinstance(port_config, dict) or not port_config:
        raise ValueError("recurrence checkpoint has no exact port configuration")
    if not torch.isfinite(control_rows).all():
        raise ValueError("recurrence checkpoint control_rows contains non-finite values")
    if control_head_rows is not None and not torch.isfinite(control_head_rows).all():
        raise ValueError("recurrence checkpoint control_head_rows contains non-finite values")
    parent_port_config = dict(parent_manifest.get("port_config") or {})
    # The direct recurrence may adjust capacity and transport, but cannot change
    # the read/stop architecture reconstructed from its clean-head anchor.
    for field in ("cutoff", "profile", "stop_source", "stop_position", "read_adapter",
                  "read_adapter_norm_eps"):
        if field in parent_port_config and port_config.get(field) != parent_port_config[field]:
            raise ValueError("recurrence changed unsupported parent port architecture field: " + field)
    if port_config.get("profile") != "raw-token-v1":
        raise ValueError("direct recurrence export requires the parent's raw-token-v1 port")
    maximum = port_config.get("max_length")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("recurrence checkpoint port max_length must be a positive integer")

    source_code = _recurrence_code_pins(identity)
    export_options = {key: value for key, value in source_options.items()
                      if key not in {"direct_artifact", "direct_parent_proof"}}
    direct_identity = {
        "schema": CHECKPOINT_SCHEMA,
        "options": {**export_options, "heads": str(anchor_path)},
        "inputs": {str(anchor_path): anchor_sha},
        "code": source_code,
    }
    export_state = {
        **state,
        "schema": CHECKPOINT_SCHEMA,
        "identity": direct_identity,
        "student_parameters": student_parameters,
        "neuralese_projection": neuralese_projection,
        "token_choice_projection": token_choice_projection,
        "control_rows": control_rows,
        "control_head_rows": control_head_rows,
        "port_config": port_config,
    }
    provenance = {
        "source_schema": RECURRENCE_CHECKPOINT_SCHEMA,
        "source_identity_sha256": canonical_json_sha256_hex(identity),
        "source_backbone_training": backbone_training,
        "parent_direct_artifact": {
            "artifact_path": parent["artifact_path"],
            "manifest_path": parent["manifest_path"],
            "manifest_file_sha256": parent["manifest_file_sha256"],
            "manifest_sha256": parent["manifest_sha256"],
            "payload_path": parent["payload_path"],
            "payload_sha256": parent["payload_sha256"],
        },
        "migration_proof": migration_proof,
    }
    return export_state, inherited_source_proof, provenance


def export_direct_checkpoint(checkpoint_path: str | Path, output_dir: str | Path, *,
                             base_snapshot: str | Path | None, base_source_proof: dict,
                             backbone_adapter: str,
                             selection_receipt: str | Path | None = None,
                             recurrence_parent_proof: dict | None = None,
                             recurrence_parent_artifact: str | Path | None = None) -> dict:
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
    recurrence_provenance = None
    if state.get("schema") == RECURRENCE_CHECKPOINT_SCHEMA:
        state, inherited_proof, recurrence_provenance = _recurrence_export_state(
            state, checkpoint_path, checkpoint_sha, recurrence_parent_proof,
            recurrence_parent_artifact)
        if canonical_json_sha256_hex(base_source_proof) != canonical_json_sha256_hex(inherited_proof):
            raise ValueError("requested base source proof differs from the recurrence checkpoint's direct parent")
        if backbone_adapter != LFM2_ADAPTER:
            raise NotImplementedError("native recurrence export currently supports only LFM2")
    elif recurrence_parent_proof is not None:
        raise ValueError("recurrence parent proof was supplied for a non-recurrence checkpoint")
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

    if recurrence_provenance is None:
        controls = {
            "control_rows": anchor["control_rows"].detach().cpu().contiguous(),
            "control_head_rows": (None if anchor.get("control_head_rows") is None else
                                  anchor["control_head_rows"].detach().cpu().contiguous()),
            "port_config": dict(anchor.get("port_config") or {}),
        }
        control_state_source = "clean-head-anchor"
    else:
        controls = {
            "control_rows": state["control_rows"].detach().cpu().contiguous(),
            "control_head_rows": (None if state.get("control_head_rows") is None else
                                  state["control_head_rows"].detach().cpu().contiguous()),
            "port_config": dict(state["port_config"]),
        }
        if controls["control_rows"].shape != anchor["control_rows"].shape:
            raise ValueError("recurrence control_rows shape differs from the pinned clean-head anchor")
        anchor_head_rows = anchor.get("control_head_rows")
        if ((controls["control_head_rows"] is None) != (anchor_head_rows is None) or
                (anchor_head_rows is not None and controls["control_head_rows"].shape != anchor_head_rows.shape)):
            raise ValueError("recurrence control_head_rows layout differs from the pinned clean-head anchor")
        control_state_source = "native-recurrence-checkpoint"
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
                                "schema": (recurrence_provenance["source_schema"]
                                           if recurrence_provenance is not None else state["schema"]),
                                "step": int(state["step"]),
                                "identity_sha256": (recurrence_provenance["source_identity_sha256"]
                                                    if recurrence_provenance is not None else
                                                    canonical_json_sha256_hex(identity)),
                                "export_identity_sha256": canonical_json_sha256_hex(identity),
                                "base_identity_in_checkpoint": recurrence_provenance is not None,
                                **({"source_identity_sha256": recurrence_provenance["source_identity_sha256"],
                                   "source_schema": recurrence_provenance["source_schema"],
                                   "source_backbone_training": recurrence_provenance["source_backbone_training"],
                                   "parent_direct_artifact": recurrence_provenance["parent_direct_artifact"],
                                   "parent_migration_proof": recurrence_provenance["migration_proof"]}
                                  if recurrence_provenance is not None else {})},
        "selection_receipt": selection,
        "base_binding": source_binding,
        "anchor": {"path": str(anchor_path), "sha256": inputs[str(anchor_path)]},
        "identity": identity,
        "projection": {"neuralese": projection_config,
                       "token_choice": (None if payload["token_choice_projection"] is None else
                                        {**projection_config, "present": True})},
        "projection_implementation": projection_code,
        "port_config": payload["port_config"],
        "control_state_source": control_state_source,
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
    if base_snapshot is None:
        raise ValueError("a direct artifact restores into its pinned base: pass the base snapshot "
                         "(--direct-base-snapshot, the manifest's base_binding model path)")
    artifact_dir = Path(artifact_dir).resolve(strict=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest.pop("manifest_sha256", None)
    if manifest.get("schema") != ARTIFACT_SCHEMA or claimed != canonical_json_sha256_hex(manifest):
        raise ValueError("direct artifact manifest schema/hash mismatch")
    manifest["manifest_sha256"] = claimed
    if manifest.get("status") != "direct-writer-pending":
        raise ValueError("direct artifact status is not supported by this loader")
    parent_port = dict(manifest.get("port_config") or {})
    if parent_port.get("profile") not in (None, "raw-token-v1"):
        raise ValueError("direct artifact loader currently supports the raw-token-v1 port only")
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
    from ..model.neuralese_projection import NeuraleseProjection, TokenChoiceProjection
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
        # The latent policy names native module paths; the export's coverage declaration (like the checkpoint's
        # student_parameters) carries the trainer's "backbone." prefix. Compare in the declaration's naming.
        declared = {name for name, _ in latent_backbone_parameters(engine.backbone)}
        declared_prefixed = sorted(prefix + name for name in declared)
        # Restore into the original BF16 Mellum. A run-v1 trainer ordering bug
        # omitted only q/k/v/o attention projections; their values remain the
        # unchanged source-base values. Later direct runs may save the complete
        # latent policy. Any other partial/mismatched name set is rejected.
        allowed_attention = {name for name in declared if any(
            name.endswith(f".self_attn.{projection}_proj.weight")
            for projection in ("q", "k", "v", "o"))}
        coverage = manifest.get("training_parameter_coverage") or {}
        missing = declared - set(native_parameters)
        if declared_prefixed != coverage.get("expected_latent_policy_names"):
            raise ValueError("Maple latent policy parameter names differ from the export coverage declaration")
        if sorted(prefix + name for name in missing) != coverage.get("missing_declared_names"):
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
        config = {k: v for k, v in manifest['projection']['token_choice'].items() if k != 'present'}
        token_choice = TokenChoiceProjection(width, **config)
        token_choice.load_state_dict(payload['token_choice_projection'], strict=True)
        token_choice.to(device=device).eval()
        for parameter in token_choice.parameters():
            parameter.requires_grad_(False)
    expected_head_controls = payload["control_head_rows"]
    actual_head_controls = getattr(engine.backbone, "control_head_rows", None)
    if (expected_head_controls is None) != (actual_head_controls is None):
        raise ValueError("loaded clean-base control-head layout differs from export")
    if (payload["control_rows"].shape != engine.backbone.control_rows.shape or
            (expected_head_controls is not None and expected_head_controls.shape != actual_head_controls.shape)):
        raise ValueError("exported control tensor shapes differ from the loaded base")
    if (not torch.isfinite(payload["control_rows"]).all() or
            (expected_head_controls is not None and not torch.isfinite(expected_head_controls).all())):
        raise ValueError("exported control tensors contain non-finite values")
    with torch.no_grad():
        engine.backbone.control_rows.copy_(payload["control_rows"].to(engine.backbone.control_rows))
        if expected_head_controls is not None:
            actual_head_controls.copy_(expected_head_controls.to(actual_head_controls))
    port_config = dict(manifest["port_config"])
    read_adapter = getattr(engine.heads, "read_adapter", None)
    actual_structure = {
        "cutoff": engine.heads.cutoff,
        "profile": engine.heads.profile,
        "stop_source": engine.heads.stop_source,
        "stop_position": engine.heads.stop.use_position,
        "read_adapter": "full-residual-v1" if read_adapter is not None else None,
        "read_adapter_norm_eps": float(read_adapter.norm.eps) if read_adapter is not None else None,
    }
    for field, actual in actual_structure.items():
        if field in port_config and actual != port_config[field]:
            raise ValueError("export port architecture differs from clean-head anchor: " + field)
    if "content_transport" in port_config:
        engine.heads.set_content_transport(port_config["content_transport"])
    maximum = port_config.get("max_length", engine.heads.max_length)
    from ..model.capacity import set_write_capacity
    set_write_capacity(engine.heads, maximum, token_stopping=True)
    engine.max_block = maximum
    actual_port_config = engine.heads.port_config()
    actual_port_config.update({"cutoff": engine.heads.cutoff,
                               "max_length": engine.heads.max_length})
    if actual_port_config != port_config:
        raise ValueError("restored PortHeads control metadata differs from direct export")
    # Shared Engine dispatch can opt into the direct writer only after that
    # writer has been independently reviewed; until then the manifest stays
    # explicitly pending and ordinary HF generation remains a separate path.
    engine.neuralese_projection = neuralese
    engine.token_choice_projection = token_choice
    engine.direct_artifact_manifest = manifest
    return DirectServingModel(engine, neuralese, token_choice, manifest)
