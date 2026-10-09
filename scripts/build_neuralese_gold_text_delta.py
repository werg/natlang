#!/usr/bin/env python3
"""Derive source-reviewed native-action additions to an existing gold-text packet.

The base packet is copied byte-for-byte. A hash-bound source selection receipt
specifies the exact delta IDs; it does not itself grant cohort admission. Delta records are rendered with the
shared ``gold_text_rows`` contract and the caller-selected tokenizer/package
snapshot; exact existing documents are accounted for without rewriting the base.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.data.context_refs import context_ref_count
from root_derived_writer_admission import admitted_root_derived_writer_rows
sys.path.insert(0, str(ROOT / "scripts"))
from root_integration_adoption import root_integration_adoption_bindings
from root_admission_scope import no_new_world_credit
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_hex as sha  # noqa: E402
from natlang_neuralese.common.jsonio import canonical_json_str as canonical  # noqa: E402


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def load_pinned_tokenizer(tokenizer_path):
    """Load the locally pinned tokenizer, including current tokenizers-only snapshots.

    Some recent model snapshots identify their fast backend as `TokenizersBackend`,
    which older Transformers releases do not register with AutoTokenizer. Falling
    back to the serialized tokenizer.json through PreTrainedTokenizerFast preserves
    the exact backend; the renderer fingerprint check below still has to match.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    path = Path(tokenizer_path)
    config_path = path / "tokenizer_config.json"
    backend_path = path / "tokenizer.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    expected_class = config.get("tokenizer_class")

    def verify(tokenizer):
        actual_class = type(tokenizer).__name__
        if expected_class == "TokenizersBackend":
            # Transformers 5.x may materialize this serialized backend through
            # either its named facade or the generic fast-tokenizer wrapper.
            # In both cases the loaded backend must be exactly the pinned JSON.
            if actual_class not in {"TokenizersBackend", "PreTrainedTokenizerFast"}:
                raise ValueError(f"tokenizer class mismatch: expected TokenizersBackend, got {actual_class}")
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is None or not callable(getattr(backend, "to_str", None)) or not backend_path.is_file():
                raise ValueError("TokenizersBackend requires its exact serialized tokenizer.json backend")
            try:
                loaded_backend = json.loads(backend.to_str())
                pinned_backend = json.loads(backend_path.read_text())
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("TokenizersBackend serialized backend is invalid") from exc
            if loaded_backend != pinned_backend:
                raise ValueError("loaded tokenizer backend differs from pinned tokenizer.json")
        elif expected_class and actual_class != expected_class:
            raise ValueError(f"tokenizer class mismatch: config declares {expected_class}, loader returned {actual_class}")
        return tokenizer

    try:
        loaded = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except ValueError as exc:
        if config.get("tokenizer_class") != "TokenizersBackend" or "TokenizersBackend" not in str(exc):
            raise
        if not backend_path.is_file():
            raise
        kwargs = {key: config[key] for key in (
            "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token",
            "mask_token", "additional_special_tokens", "clean_up_tokenization_spaces",
            "model_max_length") if config.get(key) is not None}
        template_path = path / "chat_template.jinja"
        if config.get("chat_template") is not None:
            kwargs["chat_template"] = config["chat_template"]
        elif template_path.is_file():
            kwargs["chat_template"] = template_path.read_text()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(backend_path), **kwargs)
        for name in ("padding_side", "truncation_side", "legacy", "spaces_between_special_tokens"):
            if name in config:
                setattr(tokenizer, name, config[name])
        return verify(tokenizer)
    return verify(loaded)


def read_records(path):
    out = []
    with path.open("rb") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            row["_source_record_sha256"] = sha(line.rstrip(b"\r\n"))
            out.append(row)
    return out

def provider_context_occurrences(value, block_id):
    """Compatibility name for the shared typed-marker-only count."""
    return context_ref_count(value, block_id)

def extract_combinator_definition(source_text: str, name: str) -> tuple[str, str]:
    """Read exact type/text literals from a frozen COMBINATORS property."""
    match = re.search(rf"(?ms)^\s*{re.escape(name)}\s*:\s*\{{(?P<body>[^}}]*)\}}", source_text)
    if match is None:
        raise ValueError(f"frozen COMBINATORS.{name} definition was not found")
    values = {}
    for field in ("type", "text"):
        literal = re.search(rf"\b{field}\s*:\s*(?P<literal>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*(?:[,}}]|$)",
                            match.group("body"))
        if literal is None:
            raise ValueError(f"frozen COMBINATORS.{name}.{field} is not a string literal")
        try:
            value = ast.literal_eval(literal.group("literal"))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"frozen COMBINATORS.{name}.{field} is not a supported quoted string") from exc
        if not isinstance(value, str):
            raise ValueError(f"frozen COMBINATORS.{name}.{field} is not a string")
        values[field] = value
    return values["type"], values["text"]

def validate_configured_read_context(receipt: dict, definition_type: str, definition_text: str) -> None:
    """Require a provider-expanded function body to equal the pinned runtime source."""
    block = receipt.get("block") or {}
    readout = receipt.get("readout") or {}
    definition = receipt.get("definition") or {}
    block_id, revision = block.get("id"), definition.get("revision")
    body = block.get("body")
    if (not isinstance(block_id, str) or not block_id.startswith("nz1_")
            or not isinstance(revision, str) or not block_id[4:].startswith(revision)
            or definition.get("id") != "nz-fn:" + block_id
            or block.get("type") != f"Neuralese<{definition_type}>"
            or not isinstance(body, str) or body != definition_text
            or sha(body.encode("utf-8")) != block.get("body_sha256")
            or readout.get("schema") != "natlang.text-template-readout/1"
            or readout.get("call") != "return_result" or readout.get("value") != "decode"
            or readout.get("value_type") != "string"
            or readout.get("read_body_id") != block_id
            or readout.get("read_source_sha256") != sha(definition_text.encode("utf-8"))
            or any(readout.get(flag) is not False for flag in
                   ("learned_vectors", "qualification_certificate", "training_admission"))):
        raise ValueError("configured read definition differs from its exact frozen-runtime source")

def pinned_runtime_manifest(source_approval: dict, repo_root: Path) -> Path | None:
    """Resolve one exact runtime manifest through hash-pinned approval inputs.

    A root review may pin the runtime directly, or pin a materialization receipt
    whose converter provenance binds the runtime path and bytes. Only those
    hash-verified input documents are followed; the selected runtime manifest
    itself must also match its declared digest.
    """
    root = repo_root.resolve()
    candidates: dict[Path, str] = {}

    def add_runtime(raw_path, expected_sha, *, source_label):
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise ValueError(f"malformed runtime-manifest provenance in {source_label}")
        path = Path(raw_path)
        if path.is_absolute():
            raise ValueError(f"runtime-manifest path must be repository-relative in {source_label}")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"pinned runtime manifest is missing or outside repository in {source_label}")
        actual_sha = sha_file(resolved)
        if actual_sha != expected_sha:
            raise ValueError(f"pinned runtime manifest hash mismatch in {source_label}")
        candidates[resolved] = actual_sha

    def visit(value, source_label):
        if isinstance(value, dict):
            # A digest without a path is not a binding. Plans commonly record
            # runtime_manifest_sha256 while the path is held in a separate
            # field (or the runtime is pinned through a replay receipt).
            if "runtime_manifest_path" in value:
                add_runtime(value.get("runtime_manifest_path"), value.get("runtime_manifest_sha256"),
                            source_label=source_label)
            for child in value.values():
                visit(child, source_label)
        elif isinstance(value, list):
            for child in value:
                visit(child, source_label)

    for raw_path, pin in (source_approval.get("input_pins") or {}).items():
        if not isinstance(pin, dict) or not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        if path.is_absolute():
            raise ValueError("root per-action input paths must be repository-relative")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            continue  # the root admission validator reports missing or mismatched pins first
        if sha_file(resolved) != pin.get("sha256"):
            raise ValueError(f"root per-action input pin changed while resolving runtime provenance: {raw_path}")
        if resolved.name == "frozen-runtime.json":
            add_runtime(raw_path, pin.get("sha256"), source_label=raw_path)
        elif resolved.suffix == ".json":
            try:
                visit(json.loads(resolved.read_text(encoding="utf-8")), raw_path)
            except json.JSONDecodeError:
                continue

    if len(candidates) > 1:
        raise ValueError("root per-action inputs bind more than one runtime manifest")
    return next(iter(candidates), None)

def bind_exact_provider_contexts(records, source_approval, repo_root):
    """Project authenticated provider reads into the renderer as context only.

    Same-run writer-produced blocks retain their existing graph-bound path.
    Configured function definitions use the pinned runtime's read implementation
    and are never represented as a writer event or recurrence edge.
    """
    bindings = []
    runtime_manifest = pinned_runtime_manifest(source_approval, repo_root)
    runtime_root = runtime_manifest.parent if runtime_manifest else None
    runtime_data = json.loads(runtime_manifest.read_text()) if runtime_manifest else {}
    combinator_rel = "src/neuralese/combinators.ts"
    combinator_hash = (runtime_data.get("files") or {}).get(combinator_rel)
    combinator_path = runtime_root / combinator_rel if runtime_root else None
    if combinator_path and (not combinator_path.is_file() or sha_file(combinator_path) != combinator_hash):
        raise ValueError("pinned runtime read-definition source does not match its frozen manifest")
    for record in records:
        source_ref = record.get("source_ref") or {}
        receipts = source_ref.get("provider_expanded_read_contexts") or []
        metadata = []
        record_bindings = []
        context_omissions = []
        for receipt in receipts:
            if receipt.get("origin") not in {"same-run-producer", "configured-function-definition"}: continue
            block, write = receipt.get("block") or {}, receipt.get("producer_write") or {}
            read, turn = receipt.get("block_read") or {}, receipt.get("model_turn") or {}
            block_id, body, body_hash = block.get("id"), block.get("body"), block.get("body_sha256")
            expected_read_name = None
            for existing in ((record.get("neuralese_conversion") or {}).get("external_context_inputs") or []):
                if isinstance(existing, dict) and existing.get("block_id") == block_id:
                    expected_read_name = existing.get("target_write_name")
                    if not isinstance(expected_read_name, str):
                        expected_read_name = "soft-state:" + block_id
                    break
            occurrences = context_ref_count(record.get("messages") or [], block_id, expected_read_name)
            if occurrences == 0: continue
            # A root-admitted derived target has a separate ID/target. Provider
            # reads still bind to the original sampled assistant action that
            # supplied the visible message context.
            derived = record.get("derived_target") or {}
            source_action_target = (derived.get("original_target")
                                    if derived.get("schema") == "natlang.root-admitted-derived-text-writer-target/1"
                                    else record.get("target"))
            target_hash = sha(canonical(source_action_target or {}).encode())
            required = {
                "invocation_id": source_ref.get("invocation_id"),
                "parent_invocation_id": receipt.get("parent_invocation_id"),
                "source_row_sha256": source_ref.get("source_row_sha256"),
                "transport_provenance_sha256": receipt.get("transport_provenance_sha256"),
                "source_request_sha256": receipt.get("source_request_sha256"),
                "source_response_sha256": receipt.get("source_response_sha256"),
                "source_action_target_sha256": receipt.get("source_action_target_sha256"),
                "source_trajectory_index": receipt.get("source_trajectory_index"),
                "read_node": read.get("node"), "model_turn_node": turn.get("node"),
                "producer_write_node": write.get("node"),
            }
            host_result = source_ref.get("host_result_capture") or {}
            host_capture = host_result.get("capture") or {}
            # A missing parent is meaningful for a top-level invocation, but
            # only accept it when the materializer's independent terminal
            # host capture proves this exact call was complete and parentless.
            # This does not infer a parent for nested calls or relax any of the
            # request, response, target, trace, or graph bindings below.
            verified_root_invocation = (
                required["parent_invocation_id"] is None
                and source_ref.get("parent_invocation_id") is None
                and host_capture.get("call_id") == required["invocation_id"]
                and host_capture.get("capture_kind") == "invocation_output"
                and host_capture.get("parent_call_id") is None
                and host_capture.get("complete") is True
                and host_result.get("validation", {}).get("valid") is True
            )
            configured_definition = receipt.get("origin") == "configured-function-definition"
            expected_schema = ("natlang.provider-expanded-read-context/1" if configured_definition
                               else "natlang.provider-expanded-read-context/2")
            required_hashes = (receipt.get("source_row_sha256"), receipt.get("trace_sha256"),
                               receipt.get("transport_provenance_sha256"), receipt.get("raw_request_sha256"),
                               receipt.get("rendered_request_sha256"), receipt.get("source_request_sha256"),
                               receipt.get("source_response_sha256"), receipt.get("source_action_target_sha256"))
            common_invalid = (receipt.get("schema") != expected_schema or not isinstance(block_id, str)
                    or not block_id.startswith("nz1_") or not isinstance(body, str)
                    or sha(body.encode("utf-8")) != body_hash
                    or any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
                           for v in required_hashes)
                    or receipt.get("source_row_sha256") != required["source_row_sha256"]
                    or receipt.get("invocation_id") != required["invocation_id"]
                    or (required["parent_invocation_id"] is None and not verified_root_invocation)
                    or any(not isinstance(v, str) or not v for k, v in required.items()
                           if k not in {"source_trajectory_index", "parent_invocation_id"}
                           and not (configured_definition and k == "producer_write_node"))
                    or not isinstance(required["source_trajectory_index"], int)
                    or receipt.get("source_action_target_sha256") != target_hash
                    or receipt.get("source_response_sha256") != (record.get("decision") or {}).get("source_raw_response_sha256")
                    or receipt.get("learned_vectors") is not False
                    or receipt.get("qualification_certificate") is not False
                    or receipt.get("training_admission") is not False
                    or receipt.get("context_occurrences") != occurrences)
            if common_invalid:
                raise ValueError(f"{record.get('id')}: malformed or mismatched exact provider context receipt {block_id}")
            if configured_definition:
                definition = receipt.get("definition") or {}
                readout = receipt.get("readout") or {}
                expected_definition = "nz-fn:" + str(block_id)
                if runtime_manifest is None or not combinator_hash or definition.get("id") != expected_definition:
                    raise ValueError(f"{record.get('id')}: configured read definition lacks an exact frozen-runtime binding")
                runtime_type, runtime_text = extract_combinator_definition(
                    combinator_path.read_text(), "read")
                validate_configured_read_context(receipt, runtime_type, runtime_text)
                if (read.get("kind") != "block_read" or read.get("call_id") != required["invocation_id"]
                        or read.get("block") != block_id or turn.get("kind") != "model_turn"
                        or turn.get("call_id") != required["invocation_id"]
                        or not any(isinstance(edge, dict) and edge.get("node") == read.get("node")
                                   and edge.get("block") == block_id for edge in turn.get("inputs", []))
                        or receipt.get("producer_write") is not None
                        or not isinstance(read.get("node"), str) or not isinstance(turn.get("node"), str)):
                    raise ValueError(f"{record.get('id')}: configured read trace is not an exact context-only read")
                item = {"schema": "natlang.external-context-input/1",
                        "origin": "runtime-definition-context-only", "block_id": block_id,
                        "type": block["type"], "body_sha256": body_hash,
                        "invocation_id": required["invocation_id"],
                        "source_row_sha256": required["source_row_sha256"],
                        "trace_sha256": receipt.get("trace_sha256"),
                        "transport_provenance_sha256": required["transport_provenance_sha256"],
                        "raw_request_sha256": receipt.get("raw_request_sha256"),
                        "rendered_request_sha256": receipt.get("rendered_request_sha256"),
                        "source_request_sha256": required["source_request_sha256"],
                        "source_response_sha256": required["source_response_sha256"],
                        "source_trajectory_index": required["source_trajectory_index"],
                        "read_node": required["read_node"], "model_turn_node": required["model_turn_node"],
                        "runtime_manifest_sha256": sha_file(runtime_manifest),
                        "runtime_definition_source": combinator_rel,
                        "runtime_definition_source_sha256": combinator_hash,
                        "runtime_definition_id": definition["id"],
                        "context_occurrences": occurrences,
                        "writer_target_selected": False, "learner_representation": "runtime-definition-context-only",
                        "learned_vectors": False, "qualification_certificate": False,
                        "training_admission": False}
                record_bindings.append({"record_id": record.get("id"), **item})
                metadata.append(item)
                continue
            if (receipt.get("writer_target_selected") is False
                    and receipt.get("writer_witness") is None
                    and block.get("type") == "Neuralese<string>"):
                # Keep the authenticated native action available to native
                # review, but quarantine its text row because the reader edge
                # cannot be grounded to a validated producer witness. This is
                # a row-scoped omission; it must not abort unrelated records.
                context_omissions.append({
                    "reason": "missing_authenticated_writer_witness",
                    "block_id": block_id,
                    "body_sha256": body_hash,
                    "invocation_id": required["invocation_id"],
                    "source_row_sha256": required["source_row_sha256"],
                    "trace_sha256": receipt.get("trace_sha256"),
                    "transport_provenance_sha256": required["transport_provenance_sha256"],
                    "source_request_sha256": required["source_request_sha256"],
                    "source_response_sha256": required["source_response_sha256"],
                })
                continue
            if (receipt.get("writer_target_selected") is not False
                    or not isinstance(receipt.get("writer_witness"), dict)
                    or block.get("type") != "Neuralese<string>"
                    or (receipt.get("additional_read_turn_pairs") is not None
                        and not isinstance(receipt.get("additional_read_turn_pairs"), list))):
                raise ValueError(f"{record.get('id')}: malformed same-run provider context receipt {block_id}")
            item = {
                "schema": "natlang.external-context-input/1", "origin": "same-run-producer",
                "block_id": block_id, "type": block["type"], "body_sha256": body_hash,
                "invocation_id": required["invocation_id"], "source_row_sha256": required["source_row_sha256"],
                "trace_sha256": receipt.get("trace_sha256"),
                "transport_provenance_sha256": required["transport_provenance_sha256"],
                "raw_request_sha256": receipt.get("raw_request_sha256"),
                "rendered_request_sha256": receipt.get("rendered_request_sha256"),
                "source_request_sha256": required["source_request_sha256"],
                "source_response_sha256": required["source_response_sha256"],
                "source_action_target_sha256": required["source_action_target_sha256"],
                "source_trajectory_index": required["source_trajectory_index"],
                "read_node": required["read_node"], "model_turn_node": required["model_turn_node"],
                "producer_write_node": required["producer_write_node"],
                "producer_call_id": write.get("call_id"),
                "parent_invocation_id": required["parent_invocation_id"],
                "writer_source_class": receipt.get("writer_source_class"),
                "writer_witness": receipt.get("writer_witness"),
                "additional_read_nodes": [pair.get("block_read", {}).get("node")
                                           for pair in receipt.get("additional_read_turn_pairs", [])],
                "additional_model_turn_nodes": [pair.get("model_turn", {}).get("node")
                                                 for pair in receipt.get("additional_read_turn_pairs", [])],
                "context_occurrences": occurrences, "writer_target_selected": receipt["writer_target_selected"],
                "learner_representation": "typed-read-from-authenticated-runtime-writer-event-context-only",
                "learned_vectors": False, "qualification_certificate": False, "training_admission": False,
            }
            metadata.append(item)
            record_bindings.append({"record_id": record.get("id"), **item,
                                    "provider_body_sha256": body_hash,
                                    "provider_source_writer_call_id": write.get("call_id"),
                                    "provider_source_writer_class": receipt.get("writer_source_class")})
        if context_omissions:
            record["_text_context_omissions"] = context_omissions
            bindings.append({"record_id": record.get("id"),
                             "status": "omitted_unbound_provider_context",
                             "omissions": context_omissions})
        else:
            bindings.extend(record_bindings)
        existing = ((record.get("neuralese_conversion") or {}).get("external_context_inputs") or [])
        if metadata and not context_omissions:
            record["neuralese_conversion"] = {**(record.get("neuralese_conversion") or {}),
                "external_context_inputs": [*existing, *metadata]}
    return bindings

def signature(row):
    value = [row["split"], row["text"], row["token_ids"], row["supervised_suffix_start"]]
    return sha(canonical(value).encode())

def content_signature(row):
    value = [row["text"], row["token_ids"], row["supervised_suffix_start"]]
    return sha(canonical(value).encode())

def accumulate_known(base_value, increment):
    """Accumulate a tracked count, preserving None when the base inventory is unknown."""
    return None if base_value is None else base_value + increment

def extend_known(base_value, additions):
    """Extend a tracked list, preserving None when the base inventory is unknown."""
    return None if base_value is None else [*base_value, *additions]

def anchor_qualification(anchor, rendered, omissions, helper_receipt):
    """Qualify the held base anchor independently from selected delta rows."""
    anchor_id = anchor.get("id")
    anchor_omission = next((item for item in omissions if item.get("id") == anchor_id), None)
    if helper_receipt.get("excluded_train_exact_held_complete_documents") != 0:
        return {"qualified": False, "reason": "train_document_collides_with_held_content",
                "excluded_train_exact_held_complete_documents": helper_receipt.get(
                    "excluded_train_exact_held_complete_documents")}
    if anchor_omission is not None:
        return {"qualified": False, "reason": "anchor_omitted_by_shared_text_renderer",
                "omission": anchor_omission}
    rendered_ids = {source_id for item in rendered
                    for source_id in item.get("source_record_ids", [item.get("id")])}
    if anchor_id not in rendered_ids:
        return {"qualified": False, "reason": "anchor_not_present_in_rendered_rows"}
    return {"qualified": True, "reason": "rendered_as_held_test_anchor"}

def anchor_complexity(record):
    """Try self-contained ordinary held records before reference-heavy anchors."""
    def refs(value):
        if isinstance(value, dict):
            own = isinstance(value.get("type"), str) and value["type"] in {"read", "soft", "neuralese"}
            return int(own) + sum(refs(child) for child in value.values())
        if isinstance(value, list): return sum(refs(child) for child in value)
        return 0
    messages, target = record.get("messages") or [], record.get("target") or {}
    return (refs(messages) + refs(target), len(canonical(messages)) + len(canonical(target)), record.get("id", ""))

def adopted_text_prefix_metadata(binding, *, root: Path = ROOT):
    """Read tokenizer/count metadata from a root-adopted assembled prefix.

    Consolidated assemblies intentionally have no mutable sibling receipt.json.
    Their pinned integration review binds the assembly manifest, which in turn
    binds the exact text output and tokenizer fingerprint. Derive the converter
    inputs from those reviewed facts instead of manufacturing a sidecar receipt.
    """
    integration_path = binding["integration_receipt"]
    integration = json.loads(integration_path.read_text())
    entry = integration.get("artifacts", {}).get("assembly_manifest", {})
    rel, expected = entry.get("path"), entry.get("sha256")
    if not isinstance(rel, str) or not isinstance(expected, str):
        raise ValueError("root-adopted text prefix lacks a pinned assembly manifest")
    manifest_path = (root / rel).resolve()
    if not manifest_path.is_relative_to(root.resolve()) or not manifest_path.is_file() or sha_file(manifest_path) != expected:
        raise ValueError("root-adopted assembly manifest is missing, outside the repository, or hash-mismatched")
    manifest = json.loads(manifest_path.read_text())
    adopted_text = binding["artifacts"]["text"]
    text_output = manifest.get("outputs", {}).get("text.jsonl", {})
    text_facts = manifest.get("composition", {}).get("text", {})
    counts = binding.get("counts", {}).get("text", {})
    if (text_output.get("sha256") != adopted_text["sha256"]
            or text_output.get("bytes") != adopted_text["bytes"]
            or not isinstance(text_facts.get("tokenizer_sha256"), str)
            or not all(isinstance(counts.get(k), int) for k in ("total", "train", "test"))
            or counts["total"] != counts["train"] + counts["test"]
            or any(manifest.get("composition", {}).get("text", {}).get(k) != counts[lookup]
                   for k, lookup in (("rows", "total"), ("train", "train"), ("test", "test")))):
        raise ValueError("root-adopted assembly manifest text hash, tokenizer, or counts do not match its review")
    text_counts = json.loads(integration_path.read_text()).get("counts", {}).get("text", {})
    return {
        "schema": "natlang.root-adopted-text-prefix-metadata/1",
        "documents": counts["total"], "train_documents": counts["train"],
        "test_documents": counts["test"], "tokenizer_sha256": text_facts["tokenizer_sha256"],
        "status": "derived from exact root-adopted integration review and pinned assembly manifest",
        # The adoption review proves delta coverage, not the assembly's full
        # historical omission inventory. Keep that distinction explicit.
        "renderer_code": {}, "omitted_records": None, "unresolved_omissions": None,
        "adopted_delta_omissions": text_counts.get("omissions_for_delta"),
        "duplicate_same_split_documents_deduplicated": None,
        "excluded_train_exact_held_complete_documents": None,
        "source_assembly_manifest": {"path": str(manifest_path), "sha256": expected},
    }

def resolve_base_text_prefix_metadata(binding, receipt_path: Path, *, root: Path = ROOT):
    """Use root-adopted manifest facts, checking any legacy sibling receipt agrees."""
    sibling = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    if binding is None:
        if sibling is None:
            raise ValueError("base text prefix requires its receipt.json or a verified root-adopted assembly manifest")
        return sibling
    integration = json.loads(binding["integration_receipt"].read_text())
    has_assembly_manifest = isinstance(integration.get("artifacts", {}).get("assembly_manifest"), dict)
    if not has_assembly_manifest:
        # Older verified adoption bindings predate consolidated assembly
        # manifests. Continue to support their independently pinned receipt
        # path; a sibling receipt alone does not become adoption evidence.
        if sibling is None:
            raise ValueError("root-adopted prefix without an assembly manifest requires its verified legacy receipt")
        return sibling
    adopted = adopted_text_prefix_metadata(binding, root=root)
    if sibling is None:
        return adopted
    fields = {"documents": "documents", "train_documents": "train_documents",
              "test_documents": "test_documents", "tokenizer_sha256": "tokenizer_sha256"}
    if any(sibling.get(left) != adopted.get(right) for left, right in fields.items()):
        raise ValueError("base sibling receipt count/tokenizer metadata conflicts with root-adopted assembly manifest")
    # Only the four fields above are adoption-bound. Keep an old sibling
    # receipt available as explicitly unbound metadata; never merge its
    # auxiliary claims into the root-adopted facts.
    return {**adopted, "legacy_sibling_receipt_metadata": sibling}

def selected_delta_omissions(omissions, delta_ids):
    return [item for item in omissions if item.get("id") in delta_ids]

def admission_source_groups_match(admission, row):
    """Require exact source-group binding across receipt schema variants."""
    groups = admission.get("source_groups")
    if groups is None and isinstance(admission.get("source_group"), str):
        groups = [admission["source_group"]]
    return isinstance(groups, list) and groups == row.get("source_groups", [])

def admitted_root_per_action_rows(receipt, *, delta_ids=None, root: Path = ROOT):
    """Validate the mixed-disposition root review and return only admitted rows.

    This schema deliberately retains failed, duplicate and held rows beside the
    admitted decisions. Do not treat every receipt row as an admission.
    """
    if receipt.get("schema") != "natlang.root-per-action-training-admission/1":
        raise ValueError("unsupported root per-action admission schema")
    root = root.resolve()
    review_rel, review_sha = receipt.get("review_path"), receipt.get("review_sha256")
    if not isinstance(review_rel, str) or not isinstance(review_sha, str):
        raise ValueError("root per-action receipt lacks its reviewed selection pin")
    review_path = Path(review_rel)
    if review_path.is_absolute():
        raise ValueError("root per-action review path must be repository-relative")
    review_path = (root / review_path).resolve()
    if not review_path.is_relative_to(root) or not review_path.is_file() or sha_file(review_path) != review_sha:
        raise ValueError("root per-action review is missing or its hash does not match")
    input_pins = receipt.get("input_pins")
    if not isinstance(input_pins, dict) or not input_pins:
        raise ValueError("root per-action receipt lacks pinned review inputs")
    for rel, pin in input_pins.items():
        path = Path(rel)
        if path.is_absolute() or not isinstance(pin, dict):
            raise ValueError("root per-action input pin is malformed")
        path = (root / path).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or sha_file(path) != pin.get("sha256")
                or path.stat().st_size != pin.get("bytes")):
            raise ValueError(f"root per-action input pin is missing or mismatched: {rel}")

    rows = receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("root per-action receipt has no decision rows")
    known_decisions = {
        "admit-ordinary-native-action": "admitted_native_count",
        "hold-source-required-neuralese-reader-contract": "held_source_contract_final_count",
        "hold-ambiguous-source-read-scope": "held_ambiguous_source_read_scope_count",
        "hold-unrecorded-action": "held_unrecorded_action_count",
        "reject-action-failed": "failed_count",
        "exclude-already-admitted-case01-duplicate": "already_adopted_count",
    }
    seen, observed = set(), {field: 0 for field in known_decisions.values()}
    admitted = []
    for row in rows:
        native_id, decision = row.get("native_id"), row.get("decision")
        if not isinstance(native_id, str) or not native_id or native_id in seen:
            raise ValueError(f"root per-action receipt has a missing or duplicate native_id: {native_id!r}")
        seen.add(native_id)
        field = known_decisions.get(decision)
        if field is None:
            raise ValueError(f"root per-action receipt has an unknown decision: {decision!r}")
        flag = row.get("training_admission")
        if not isinstance(flag, bool) or flag != (decision == "admit-ordinary-native-action"):
            raise ValueError(f"root per-action admission flag conflicts with decision for {native_id}")
        observed[field] += 1
        if flag:
            if row.get("split") != "train":
                raise ValueError(f"root per-action admission is not train-only: {native_id}")
            admitted.append(row)
    if receipt.get("admitted_native_count") != len(admitted):
        raise ValueError("root per-action admitted_native_count conflicts with decision rows")
    if any(receipt.get(field, 0) != count for field, count in observed.items()):
        raise ValueError("root per-action disposition counts conflict with decision rows")
    if sum(observed.values()) != len(rows):
        raise ValueError("root per-action disposition counts do not cover every decision row")
    if (receipt.get("whole_trajectory_admission") is not False
            or receipt.get("runtime_qualification") is not False
            or receipt.get("active_gpu_inputs_changed") is not False
            or not no_new_world_credit(receipt.get("new_world_credit"))):
        raise ValueError("root per-action receipt includes an unsupported admission facet")
    approved_ids = [row["native_id"] for row in admitted]
    if delta_ids is not None and (not isinstance(delta_ids, set) or delta_ids != set(approved_ids)):
        raise ValueError("delta records do not equal root-admitted per-action IDs")
    return admitted

def append_prefix(prefix: Path, output: Path, additions: list[dict]):
    with output.open("xb") as f:
        with prefix.open("rb") as src:
            last = b""
            for block in iter(lambda: src.read(1 << 20), b""):
                f.write(block); last = block[-1:]
        if prefix.stat().st_size and last != b"\n": f.write(b"\n")
        for row in additions: f.write((canonical(row) + "\n").encode())
    if sha_prefix(output, prefix.stat().st_size) != sha_file(prefix):
        raise ValueError(f"base prefix changed: {prefix}")

def sha_prefix(path: Path, length: int) -> str:
    h = hashlib.sha256(); remaining = length
    with path.open("rb") as f:
        while remaining:
            block = f.read(min(1 << 20, remaining))
            if not block: raise ValueError("unexpected EOF during prefix check")
            h.update(block); remaining -= len(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-text", "base-provenance", "base-records", "base-root-receipt",
                 "delta-records", "pieces", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--base-pieces", type=Path,
                   help="approved base native piece catalog (defaults to native-pieces.jsonl beside --base-records)")
    p.add_argument("--source-selection", "--source-approval", dest="source_approval", required=True, type=Path,
                   help="hash-bound source/action selection; this is not root cohort admission")
    p.add_argument("--tokenizer", required=True, help="Pinned tokenizer ID or local snapshot")
    p.add_argument("--twin-of-text", type=Path,
                   help="the root-approved text whose tokenizer twin --base-text is: the receipt must bind it, and "
                        "--base-text must carry the identical document ID/split/source-group/source-record sequence")
    p.add_argument("--renderer-package-root", type=Path,
                   help="Path containing natlang_neuralese/; use the reviewed renderer snapshot")
    p.add_argument("--repo-root", type=Path,
                   help="Repository root for hash-bound receipts and artifacts; defaults to the builder's checkout")
    p.add_argument("--base-text-delta", type=Path, action="append", default=[],
                   help="previously adopted compact text additions to include in virtual prefix collision checks")
    p.add_argument("--base-text-delta-adoption", type=Path,
                   help="root adoption receipt that binds every supplied --base-text-delta")
    p.add_argument("--base-text-delta-manifest", type=Path,
                   help="immutable corpus manifest that pins the compact composition files, including each text delta")
    p.add_argument("--compact-only", action="store_true",
                   help="write selected text/provenance/coverage deltas only; do not copy full text prefixes")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise ValueError(f"output is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    repo_root = (args.repo_root or ROOT).resolve()
    package = (args.renderer_package_root or (repo_root / "training/neuralese")).resolve()
    sys.path.insert(0, str(package))
    from natlang_neuralese.data.text_corpus import gold_text_rows
    base_records = read_records(args.base_records)
    delta_records = read_records(args.delta_records)
    base_approval = json.loads(args.base_root_receipt.read_text())
    adoption_bindings = root_integration_adoption_bindings(base_approval, root=repo_root)
    base_receipt_path = args.base_text.parent / "receipt.json"
    base_prefix_receipt = resolve_base_text_prefix_metadata(adoption_bindings, base_receipt_path, root=repo_root)
    if (not isinstance(base_prefix_receipt.get("train_documents"), int)
            or not isinstance(base_prefix_receipt.get("test_documents"), int)
            or base_prefix_receipt["train_documents"] < 1
            or base_prefix_receipt["test_documents"] < 1):
        raise ValueError("root-adopted base text prefix lacks independently populated train and held splits")
    approved_text = args.twin_of_text or args.base_text
    def manifest_binds(text):  # a root admission that binds the packet's output manifest, which binds the text
        manifest = text.parent / "output-manifest.json"
        return (manifest.is_file() and base_approval.get("output_manifest_sha256") == sha_file(manifest) and
                json.loads(manifest.read_text())["outputs"]["text.jsonl"]["sha256"] == sha_file(text))
    root_admission = base_approval.get("schema") == "natlang.root-corpus-admission/1"
    root_integration_admission = base_approval.get("schema") == "natlang.root-corpus-integration-admission/1"
    root_integration_adoption = adoption_bindings is not None
    if root_admission and (not str(base_approval.get("status", "")).startswith("admitted-")
                           or base_approval.get("admission", {}).get("native_sft") is not True
                           or base_approval.get("admission", {}).get("derived_ordinary_gold_text") is not True):
        raise ValueError("base root receipt does not admit its native and derived-text facets")
    root_text = next((entry for entry in base_approval.get("files", {}).values()
                      if entry.get("path", "").endswith("text.jsonl")), None) if root_admission else None
    root_text_binds = bool(root_text and root_text.get("sha256") == sha_file(approved_text))
    integration_text_binds = bool(root_integration_admission and
        base_approval.get("qualification_scope", {}).get("native_sft_only") is True and
        base_approval.get("outputs", {}).get("text", {}).get("sha256") == sha_file(approved_text))
    adoption_text_binds = bool(root_integration_adoption and
        adoption_bindings["artifacts"]["text"]["sha256"] == sha_file(approved_text) and
        adoption_bindings["artifacts"]["provenance"]["sha256"] == sha_file(args.base_provenance) and
        adoption_bindings["artifacts"]["native"]["sha256"] == sha_file(args.base_records))
    receipt_approves_base = (base_approval.get("approved") is True or root_admission or
                             root_integration_admission or root_integration_adoption)
    receipt_binds_text = (base_approval.get("text_sha256") == sha_file(approved_text) or
                          root_text_binds or integration_text_binds or adoption_text_binds or
                          manifest_binds(approved_text))
    if not receipt_approves_base or not receipt_binds_text:
        raise ValueError("base text root receipt does not approve/bind the exact text prefix")
    prior_text_delta_adoption = None
    if args.base_text_delta:
        if not args.base_text_delta_adoption:
            raise ValueError("--base-text-delta requires --base-text-delta-adoption")
        prior_text_delta_adoption = json.loads(args.base_text_delta_adoption.read_text())
        if (prior_text_delta_adoption.get("schema") != "natlang.root-compact-delta-composition-adoption/1"
                or prior_text_delta_adoption.get("decision") != "approve-exact-union-of-separately-adopted-native-text-deltas"):
            raise ValueError("prior compact text additions lack the expected root composition adoption")
        review_rel = prior_text_delta_adoption.get("composition_review")
        review_sha = prior_text_delta_adoption.get("composition_review_sha256")
        review_path = (repo_root / review_rel).resolve() if isinstance(review_rel, str) else None
        if (not review_path or not review_path.is_relative_to(repo_root) or not review_path.is_file()
                or sha_file(review_path) != review_sha):
            raise ValueError("prior text delta adoption does not bind its composition review")
        composition_review = json.loads(review_path.read_text())
        artifact_hashes = {item.get("path"): item.get("sha256")
                           for item in composition_review.get("artifacts", [])}
        if not args.base_text_delta_manifest:
            raise ValueError("--base-text-delta requires its registered immutable corpus manifest")
        prior_manifest = json.loads(args.base_text_delta_manifest.read_text())
        prior_manifest_paths = {item.get("path"): item.get("sha256") for item in prior_manifest.get("files", [])}
        manifest_root = (repo_root / prior_manifest.get("path", "")).resolve()
        adoption_rel = str(args.base_text_delta_adoption.resolve().relative_to(manifest_root))
        if prior_manifest_paths.get(adoption_rel) != sha_file(args.base_text_delta_adoption):
            raise ValueError("registered corpus manifest does not pin the prior root composition adoption")
        for prior_delta in args.base_text_delta:
            rel = str(prior_delta.resolve().relative_to(manifest_root))
            if prior_manifest_paths.get(rel) != sha_file(prior_delta):
                raise ValueError(f"prior compact text delta is not pinned by its adoption: {rel}")
    if args.twin_of_text:
        def identity(path):  # tokenizer-independent document identity: ID, split and the admitted source records
            with path.open("r", encoding="utf-8") as f:
                return [(r["id"], r["split"], r.get("source_groups"), r.get("source_record_ids"))
                        for r in map(json.loads, filter(str.strip, f))]
        if identity(args.twin_of_text) != identity(args.base_text):
            raise ValueError("tokenizer twin does not carry the approved text's document IDs, splits, source groups "
                             "and source record IDs in order")
    source_approval = json.loads(args.source_approval.read_text())
    root_action_admission = source_approval.get("schema") == "natlang.root-selected-action-admission/1"
    root_per_action_admission = source_approval.get("schema") == "natlang.root-per-action-training-admission/1"
    root_derived_writer_admission = source_approval.get("schema") in {
        "natlang.root-derived-observed-text-writer-admission/1", "natlang.root-derived-body-admission-index/1"}
    admission_rows = source_approval.get("rows", []) if root_action_admission else []
    if root_action_admission:
        approved_ids = [item.get("native_id") for item in admission_rows]
        if not approved_ids or any(item.get("decision") != "admit-exact-selected-native-action-SFT-only"
                                   for item in admission_rows):
            raise ValueError("root action receipt has no exact native SFT-only admission rows")
        counts = source_approval.get("counts") or {}
        legacy_counts = (counts.get("native_SFT_train_actions") == len(admission_rows)
                         and counts.get("whole_trajectories") == 0)
        selected_counts = (counts.get("selected_native_actions") == len(admission_rows)
                           and counts.get("train_actions") == len(admission_rows)
                           and counts.get("test_actions") == 0
                           and counts.get("whole_trajectories") == 0
                           and counts.get("new_worlds") == 0)
        qualifications = source_approval.get("qualifications", {})
        if (not (legacy_counts or selected_counts)
                or qualifications.get("learned_writer") is not False
                or qualifications.get("recurrence") is not False
                or source_approval.get("integration", {}).get("active_GPU_inputs_changed") is not False):
            raise ValueError("root action receipt includes unsupported non-native admission facets")
    elif root_per_action_admission:
        admission_rows = admitted_root_per_action_rows(
            source_approval, delta_ids={row.get("id") for row in delta_records}, root=repo_root)
        approved_ids = [item["native_id"] for item in admission_rows]
    elif root_derived_writer_admission:
        admission_rows = admitted_root_derived_writer_rows(source_approval, args.source_approval,
                                                            delta_records=delta_records, root=repo_root)
        approved_ids = [item["native_id"] for item in admission_rows]
    else:
        approved_ids = source_approval.get("approved_row_ids")
    delta_ids = {r.get("id") for r in delta_records}
    if not isinstance(approved_ids, list) or set(approved_ids) != delta_ids or len(approved_ids) != len(delta_ids):
        raise ValueError("delta records do not equal source approval IDs")
    if root_action_admission or root_per_action_admission or root_derived_writer_admission:
        admitted_by_id = {item["native_id"]: item for item in admission_rows}
        for row in delta_records:
            admission = admitted_by_id.get(row.get("id"))
            digest = sha(json.dumps(row.get("target"), ensure_ascii=False,
                                    sort_keys=not root_derived_writer_admission, separators=(",", ":")).encode())
            if not admission or digest != admission.get("target_sha256"):
                raise ValueError(f"root action admission target binding mismatch: {row.get('id')}")
            if root_derived_writer_admission:
                group_ok = admission.get("source_group") in row.get("source_groups", [])
            elif root_per_action_admission:
                group_ok = admission_source_groups_match(admission, row)
            else:
                groups = admission.get("source_groups")
                if groups is None and isinstance(admission.get("source_group"), str): groups = [admission["source_group"]]
                group_ok = isinstance(groups, list) and groups == row.get("source_groups", [])
            if row.get("split") != admission.get("split") or not group_ok:
                raise ValueError(f"root action admission split/group mismatch: {row.get('id')}")
            # The source conversion can predate root admission and therefore carry
            # an explicit pending marker. Apply the exact root decision in memory
            # only after ID/target/split/group bindings pass; raw source bytes stay
            # pinned and unchanged.
            row["training_admission"] = {
                "kind": ("root-derived-text-writer-body-admission" if root_derived_writer_admission
                         else "root-selected-action-admission"),
                "approved": True,
                "receipt_sha256": admission.get("receipt_sha256", sha_file(args.source_approval)),
                **({"admission_index_sha256": admission["admission_index_sha256"]}
                   if root_derived_writer_admission else {}),
            }
    for rel, expected in source_approval.get("artifact_hashes", {}).items():
        bound = (repo_root / rel).resolve()
        if not bound.is_file() or sha_file(bound) != expected:
            raise ValueError(f"source approval artifact missing/hash mismatch: {rel}")
    provider_context_bindings = bind_exact_provider_contexts(delta_records, source_approval, repo_root)
    def iter_jsonl(path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    base_rows = iter_jsonl(args.base_text)
    base_keys, by_key, held_content = set(), {}, {}
    base_document_ids = set()
    for row in base_rows:
        key = signature(row)
        base_keys.add(key)
        by_key.setdefault(key, {"id": row.get("id"), "split": row["split"],
                                "source_record_ids": row.get("source_record_ids", [row.get("id")])})
        if row.get("id") in base_document_ids: raise ValueError(f"duplicate document ID in base text: {row.get('id')}")
        base_document_ids.add(row.get("id"))
        if row["split"] == "test": held_content.setdefault(content_signature(row), by_key[key])
    prior_delta_rows = []
    for prior_path in args.base_text_delta:
        for row in iter_jsonl(prior_path):
            if row.get("id") in base_document_ids:
                raise ValueError(f"prior compact text delta document ID already exists in virtual prefix: {row.get('id')}")
            base_document_ids.add(row.get("id"))
            key = signature(row)
            if key in base_keys:
                raise ValueError(f"prior compact text delta duplicates a base document signature: {row.get('id')}")
            base_keys.add(key)
            by_key[key] = {"id": row.get("id"), "split": row["split"],
                           "source_record_ids": row.get("source_record_ids", [row.get("id")])}
            if row["split"] == "test": held_content.setdefault(content_signature(row), by_key[key])
            prior_delta_rows.append(row)
    anchors = [r for r in base_records if r.get("split") == "test" and
               r.get("training_admission", {}).get("approved") is True]
    if not anchors: raise ValueError("base native records have no approved test anchor")
    anchors.sort(key=anchor_complexity)
    base_pieces_path = args.base_pieces or (args.base_records.parent / "native-pieces.jsonl")
    base_pieces = list(iter_jsonl(base_pieces_path)) if base_pieces_path.is_file() else []
    delta_pieces = list(iter_jsonl(args.pieces))
    pieces_by_name = {}
    for piece in [*base_pieces, *delta_pieces]:
        name, text = piece.get("name"), piece.get("text")
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("base or delta piece catalog contains a malformed piece")
        if name in pieces_by_name and pieces_by_name[name] != text:
            raise ValueError(f"base and delta piece text conflicts for {name}")
        pieces_by_name[name] = text
    pieces = [{"name": name, "text": text} for name, text in pieces_by_name.items()]
    tokenizer = load_pinned_tokenizer(args.tokenizer)
    rendered = omissions = provenance = helper_receipt = None
    anchor_attempts = []
    # A delta document may exactly match the test anchor. Select a deterministic
    # different held record so the shared helper's held-duplicate filter applies
    # only to the real, previously derived base packet.
    for anchor in anchors:
        rendered, helper_receipt, omissions, provenance = gold_text_rows(
            [*(r for r in delta_records if not r.get("_text_context_omissions")), anchor],
            pieces, tokenizer=tokenizer,
            require_independent_splits=False)
        qualification = anchor_qualification(anchor, rendered, omissions, helper_receipt)
        anchor_attempts.append({"id": anchor.get("id"),
                                "source_record_sha256": anchor.get("_source_record_sha256"),
                                **qualification})
        if qualification["qualified"]:
            break
    else:
        raise ValueError("no approved base test anchor survived shared rendering; see anchor-selection diagnostics")
    if helper_receipt["tokenizer_sha256"] != base_prefix_receipt["tokenizer_sha256"]:
        raise ValueError("tokenizer fingerprint differs from base packet")

    additions, coverage = [], []
    delta_omissions = selected_delta_omissions(omissions, delta_ids)
    for original in delta_records:
        for item in original.get("_text_context_omissions", []):
            delta_omissions.append({"id": original["id"], **item,
                                    "source_record_sha256": original["_source_record_sha256"]})
    omission_by_id = {r["id"]: r for r in delta_omissions}
    # Rows are already de-duplicated by the shared gold-text helper.
    for row in rendered:
        source_ids = [sid for sid in row.get("source_record_ids", [row.get("id")]) if sid in delta_ids]
        if not source_ids: continue  # the held-test anchor exists to validate split context only
        key = signature(row)
        if key in base_keys:
            existing = by_key[key]
            status = "matches_existing_v15_document"
            destination = existing
        elif row["split"] == "train" and content_signature(row) in held_content:
            status = "excluded_exact_held_document"
            destination = held_content[content_signature(row)]
        else:
            additions.append(row)
            destination = {"id": row["id"], "source_record_ids": row["source_record_ids"], "split": row["split"]}
            status = "appended_new_document"
        for rid in source_ids:
            original = next(r for r in delta_records if r["id"] == rid)
            coverage.append({"record_id": rid, "source_ids": original.get("source_ids", []),
                             "status": status, "document_signature_sha256": key,
                             "destination": destination})
    covered = {c["record_id"] for c in coverage}
    for rid, omission in omission_by_id.items():
        if rid not in delta_ids:
            continue  # the temporary held-test anchor is not part of this delta
        original = next(r for r in delta_records if r["id"] == rid)
        coverage.append({"record_id": rid, "source_ids": original.get("source_ids", []),
                         "status": ("omitted_unbound_provider_context"
                                    if omission.get("reason") == "missing_authenticated_writer_witness"
                                    else "omitted_by_shared_text_renderer"),
                         "reason": omission,
                         "source_record_sha256": original["_source_record_sha256"]})
        covered.add(rid)
    if covered != delta_ids:
        raise ValueError(f"delta record coverage incomplete: {sorted(delta_ids-covered)}")
    coverage.sort(key=lambda x: x["record_id"])
    anchor_selection = {"schema": "natlang.gold-text-delta-anchor-selection/1",
                        "selected_anchor_id": anchor.get("id"),
                        "attempts": anchor_attempts,
                        "omissions_are_not_delta_coverage": True}
    (args.out / "anchor-selection.json").write_text(json.dumps(anchor_selection, indent=2,
        ensure_ascii=False, sort_keys=True)+"\n")
    add_provenance = [x for x in provenance if x.get("id") in delta_ids]
    base_omissions = args.base_text.parent / "omissions.jsonl"
    if args.compact_only:
        (args.out / "delta-text.jsonl").write_text("".join(canonical(x)+"\n" for x in additions), encoding="utf-8")
        (args.out / "delta-provenance.jsonl").write_text("".join(canonical(x)+"\n" for x in add_provenance), encoding="utf-8")
        (args.out / "delta-omissions.jsonl").write_text("".join(canonical(x)+"\n" for x in delta_omissions), encoding="utf-8")
    else:
        append_prefix(args.base_text, args.out / "text.jsonl", additions)
        append_prefix(args.base_provenance, args.out / "provenance.jsonl", add_provenance)
        append_prefix(base_omissions, args.out / "omissions.jsonl", delta_omissions)
    coverage_bytes = "".join(canonical(x)+"\n" for x in coverage).encode()
    (args.out / "source-coverage.jsonl").write_bytes(coverage_bytes)
    context_binding_bytes = "".join(canonical(x)+"\n" for x in provider_context_bindings).encode()
    (args.out / "provider-context-bindings.jsonl").write_bytes(context_binding_bytes)
    old_receipt = base_prefix_receipt
    receipt = dict(old_receipt)
    base_same_split = sum(c["status"] == "matches_existing_v15_document" for c in coverage)
    base_held = sum(c["status"] == "excluded_exact_held_document" for c in coverage)
    prior_delta_train = sum(row["split"] == "train" for row in prior_delta_rows)
    prior_delta_test = sum(row["split"] == "test" for row in prior_delta_rows)
    virtual_base_docs = old_receipt["documents"] + len(prior_delta_rows)
    virtual_base_train = old_receipt["train_documents"] + prior_delta_train
    virtual_base_test = old_receipt["test_documents"] + prior_delta_test
    receipt.update({"status": "held compact ordinary-text delta; root integration review pending" if args.compact_only
                        else "held cumulative ordinary-text proposal; root review pending",
                    "policy": helper_receipt["policy"],
                    "rendering": helper_receipt["rendering"],
                    "supervision": helper_receipt["supervision"],
                    "ordinary_text_stage_only": True,
                    "split_qualification": {
                        "renderer_delta_call_requires_both_splits": False,
                        "base_prefix_train_documents": base_prefix_receipt["train_documents"],
                        "base_prefix_test_documents": base_prefix_receipt["test_documents"],
                        "base_root_adoption_sha256": sha_file(args.base_root_receipt),
                    },
                    "base_renderer_code": old_receipt.get("renderer_code", {}),
                    "documents": virtual_base_docs + len(additions),
                    "train_documents": virtual_base_train + sum(x["split"] == "train" for x in additions),
                    "test_documents": virtual_base_test + sum(x["split"] == "test" for x in additions),
                    "virtual_base_documents": virtual_base_docs,
                    "virtual_base_train_documents": virtual_base_train,
                    "virtual_base_test_documents": virtual_base_test,
                    "virtual_base_text_deltas": [{"path": str(path), "sha256": sha_file(path),
                                                   "rows": sum(1 for _ in iter_jsonl(path))}
                                                  for path in args.base_text_delta],
                    "prior_text_delta_adoption_sha256": sha_file(args.base_text_delta_adoption)
                        if args.base_text_delta_adoption else None,
                    "source_text_prefix_sha256": sha_file(args.base_text),
                    "source_text_prefix_bytes": args.base_text.stat().st_size,
                    "delta_source_run": str(args.delta_records.parent),
                    "delta_record_count": len(delta_records), "delta_appended_document_count": len(additions),
                    "delta_root_action_admission_overlay_records": len(admission_rows)
                    if (root_action_admission or root_per_action_admission) else 0,
                    "delta_root_derived_writer_admission_records": len(admission_rows)
                    if root_derived_writer_admission else 0,
                    "delta_hash_bound_reader_context_blocks": helper_receipt.get("hash_bound_reader_context_blocks", 0),
                    "delta_authenticated_provider_context_only_blocks": len(provider_context_bindings),
                    "provider_context_bindings_sha256": sha(context_binding_bytes),
                    "delta_capture_context_augmentation_records": sum(
                        bool(row.get("capture_context_augmentation_attestations")) for row in provenance
                        if row.get("id") in delta_ids),
                    "delta_capture_context_augmentation_messages": sum(
                        len(row.get("capture_context_augmentation_attestations", [])) for row in provenance
                        if row.get("id") in delta_ids),
                    "delta_typed_eval_finish_marker_calls": sum(
                        call.get("neuralese_code", {}).get("schema") == "natlang.neuralese-code/1"
                        for record in delta_records for call in (record.get("target") or {}).get("tool_calls", [])),
                    "omitted_records": accumulate_known(old_receipt.get("omitted_records"), len(delta_omissions)),
                    "unresolved_omissions": extend_known(old_receipt.get("unresolved_omissions"), delta_omissions),
                    "delta_omitted_records": len(delta_omissions),
                    "delta_unresolved_omissions": delta_omissions,
                    "duplicate_same_split_documents_deduplicated": accumulate_known(
                        old_receipt.get("duplicate_same_split_documents_deduplicated"),
                        helper_receipt["duplicate_same_split_documents_deduplicated"] + base_same_split),
                    "excluded_train_exact_held_complete_documents": accumulate_known(
                        old_receipt.get("excluded_train_exact_held_complete_documents"),
                        helper_receipt["excluded_train_exact_held_complete_documents"] + base_held),
                    "delta_source_coverage_sha256": sha(coverage_bytes),
                    "anchor_selection": anchor_selection,
                    "text_jsonl_sha256": sha_file(args.out / "text.jsonl") if not args.compact_only else None,
                    "provenance_jsonl_sha256": sha_file(args.out / "provenance.jsonl") if not args.compact_only else None,
                    "omissions_jsonl_sha256": sha_file(args.out / "omissions.jsonl") if not args.compact_only else None,
                    "compact_delta_outputs": {name: sha_file(args.out/name) for name in
                        ("delta-text.jsonl", "delta-provenance.jsonl", "delta-omissions.jsonl")} if args.compact_only else None,
                    "source_files": {"delta_records": sha_file(args.delta_records),
                                     "base_pieces": sha_file(base_pieces_path) if base_pieces_path.is_file() else None,
                                     "pieces": sha_file(args.pieces),
                                     "source_approval": sha_file(args.source_approval),
                                     "base_text_root_receipt": sha_file(args.base_root_receipt)},
                    "renderer_code": {"text_corpus.py": sha((package / "natlang_neuralese/data/text_corpus.py").read_bytes()),
                                      "chat.py": sha((package / "natlang_neuralese/serve/chat.py").read_bytes()),
                                      "trajectories.py": sha((package / "natlang_neuralese/train/trajectories.py").read_bytes()),
                                      "inline_instructions.py": sha((package / "natlang_neuralese/train/inline_instructions.py").read_bytes()),
                                      "delta_builder.py": sha(Path(__file__).read_bytes())},
                    "task_or_trajectory_admission_granted": False})
    if args.twin_of_text:
        receipt["tokenizer_twin_of"] = {"text_sha256": sha_file(args.twin_of_text), "path": str(args.twin_of_text)}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
    packet = {"schema": "natlang.native-gold-text-delta-proposal/1",
              "status": "held compact delta; ordinary text only; independent review required" if args.compact_only
                        else "held; ordinary text only; independent review required",
              "base_packet": {"text_sha256": receipt["source_text_prefix_sha256"],
                              "text_bytes": receipt["source_text_prefix_bytes"],
                              "document_count": virtual_base_docs,
                              "physical_document_count": old_receipt["documents"]},
              "virtual_base_text_deltas": receipt["virtual_base_text_deltas"],
              "prior_text_delta_adoption_sha256": receipt["prior_text_delta_adoption_sha256"],
              "delta_source": {"records_sha256": sha_file(args.delta_records),
                               "records": len(delta_records), "piece_sha256": sha_file(args.pieces),
                               "source_selection_sha256": sha_file(args.source_approval)},
              "coverage": {s: sum(c["status"] == s for c in coverage)
                           for s in sorted({c["status"] for c in coverage})},
              "anchor_selection_sha256": sha_file(args.out/"anchor-selection.json"),
              "outputs": {name: {"sha256": sha_file(args.out/name), "bytes": (args.out/name).stat().st_size}
                          for name in (("delta-text.jsonl", "delta-provenance.jsonl", "delta-omissions.jsonl",
                                        "source-coverage.jsonl", "provider-context-bindings.jsonl", "receipt.json", "anchor-selection.json") if args.compact_only
                                       else ("text.jsonl", "provenance.jsonl", "omissions.jsonl", "source-coverage.jsonl",
                                             "provider-context-bindings.jsonl", "receipt.json", "anchor-selection.json"))},
              "training_admission": False, "task_or_trajectory_admission": False}
    (args.out / "packet-manifest.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
    hydrated_contexts = sum(len(item.get("neuralese_context_attestations", [])) for item in provenance)
    (args.out / "README.md").write_text(
        "# Held ordinary gold-text proposal\n\n"
        "This is a text-only proposal for independent root review. "
        + ("It contains compact selected deltas; admitted prefixes are referenced by exact root receipts and are not copied. "
           if args.compact_only else "Its `text.jsonl` and `provenance.jsonl` preserve the supplied admitted text packet as exact byte prefixes. ")
        + "The selected delta was rendered with the shared `gold_text_rows` helper and the pinned tokenizer fingerprint recorded in `receipt.json`; no model generation or tools were used.\n\n"
        f"The base prefix contains {old_receipt['documents']} documents; this proposal appends {len(additions)} new documents from {len(delta_records)} selected source-reviewed records. "
        f"{hydrated_contexts} exact named Neuralese reader-context blocks were hydrated from successful, hash-bound writer target sources in the same split and source group; attestations are recorded in `provenance.jsonl`. "
        "The current delta renderer records typed eval-finish marker sidecars and separately labeled full capture context augmentations. Those augmentations come from authenticated same-invocation snapshots and do not claim the omitted text was historically provider-visible. "
        "Hydrated blocks and capture augmentations add context only; they do not create separate target rows. "
        f"{receipt['omitted_records']} records remain unresolved omissions; see `source-coverage.jsonl` and `omissions.jsonl`. "
        "The base held-test anchor qualification is recorded separately in `anchor-selection.json`; anchor omissions do not appear as selected-delta omissions. "
        "The proposal does not grant text packet admission, task/trajectory admission, model qualification, or training authorization.\n"
    )
    output_names = (["delta-text.jsonl", "delta-provenance.jsonl", "delta-omissions.jsonl", "source-coverage.jsonl",
                     "provider-context-bindings.jsonl", "receipt.json", "packet-manifest.json", "README.md", "anchor-selection.json"] if args.compact_only else
                    ["text.jsonl", "provenance.jsonl", "omissions.jsonl", "source-coverage.jsonl",
                     "provider-context-bindings.jsonl", "receipt.json", "packet-manifest.json", "README.md", "anchor-selection.json"])
    output_manifest = {"schema": "natlang.gold-text-delta-output-manifest/1",
                       "packet_manifest_sha256": sha_file(args.out/"packet-manifest.json"),
                       "outputs": {name: {"sha256": sha_file(args.out/name), "bytes": (args.out/name).stat().st_size}
                                   for name in output_names}}
    (args.out/"output-manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"base_documents": old_receipt["documents"], "appended_documents": len(additions),
                      "cumulative_documents": receipt["documents"], "delta_records": len(delta_records),
                      "coverage_statuses": {s: sum(c["status"] == s for c in coverage)
                                            for s in sorted({c["status"] for c in coverage})},
                      "tokenizer_sha256": helper_receipt["tokenizer_sha256"],
                      "text_sha256": receipt["text_jsonl_sha256"]}, indent=2))

if __name__ == "__main__": main()
