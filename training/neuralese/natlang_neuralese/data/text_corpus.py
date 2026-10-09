"""Deterministic ordinary gold-text rows for neuralese warm-up."""
from __future__ import annotations

import hashlib
import json
import copy
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .context_refs import context_ref_count, typed_neuralese_ref_count


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _direct_result_event_name(block_id: str, trajectory_id: str, call_id: str, node: str) -> str:
    event = _sha(json.dumps([trajectory_id, call_id, node], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))[:12]
    return f"soft-state:{block_id}@{event}"


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _protected_target_sidecar_equivalence(record, receipt, metadata):
    """Validate the one historical converter sidecar excluded by the current target digest."""
    from ..train.inline_instructions import validate_inline_instruction_code

    adapter = receipt.get("target_binding_adapter")
    if not isinstance(adapter, dict) or metadata.get("target_binding_adapter") != adapter:
        return False
    target = record.get("target")
    calls = target.get("tool_calls") if isinstance(target, dict) else None
    if not isinstance(calls, list):
        return False
    candidates = [(index, call) for index, call in enumerate(calls)
                  if isinstance(call, dict) and "neuralese_code" in call]
    if len(candidates) != 1:
        return False
    index, call = candidates[0]
    sidecar = call.get("neuralese_code")
    call_id = call.get("id")
    function = call.get("function") or {}
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if not isinstance(sidecar, dict) or not isinstance(call_id, str) or not isinstance(arguments, str):
        return False
    checked = validate_inline_instruction_code(arguments, sidecar)
    if not checked.valid or checked.value is None:
        return False
    writes = checked.value.writes
    sites = sidecar.get("sites")
    if not isinstance(sites, list) or len(sites) != len(writes):
        return False
    sites_by_name = {site.get("name"): site for site in sites if isinstance(site, dict)}
    if len(sites_by_name) != len(sites) or set(sites_by_name) != {write.name for write in writes}:
        return False
    code = checked.value.code
    for write in writes:
        site = sites_by_name[write.name]
        span = site.get("code_span")
        if (not isinstance(span, dict) or type(span.get("start")) is not int
                or type(span.get("end")) is not int or span["start"] != write.start - 1
                or span["end"] != write.end + 1 or span["start"] < 0
                or code[span["start"]:span["end"]] != "`" + (write.code_source or write.source) + "`"):
            return False
    if (adapter.get("schema") != "natlang.protected-target-inline-sidecar-equivalence/1"
            or adapter.get("kind") != "remove-one-validated-neuralese-code-sidecar"
            or adapter.get("call_id") != call_id
            or adapter.get("sidecar_code_sha256") != sidecar.get("code_sha256")
            or not _sha256_hex(adapter.get("sidecar_sha256"))
            or _sha(_canonical(sidecar).encode("utf-8")) != adapter.get("sidecar_sha256")):
        return False
    protected_sha = _sha(_canonical(target).encode("utf-8"))
    projection = copy.deepcopy(target)
    del projection["tool_calls"][index]["neuralese_code"]
    projection_sha = _sha(_canonical(projection).encode("utf-8"))
    return (adapter.get("protected_target_sha256") == protected_sha
            and adapter.get("materializer_target_sha256") == projection_sha
            and adapter.get("projection_sha256") == projection_sha
            and receipt.get("source_action_target_sha256") == projection_sha
            and metadata.get("source_action_target_sha256") == projection_sha)


def _authenticated_derived_semantic_text_write(record):
    """Validate the explicitly transformed pure-literal eval-to-text target view."""
    derived = record.get("derived_target")
    if isinstance(derived, dict) and derived.get("schema") == "natlang.root-admitted-derived-text-writer-target/1":
        return _authenticated_root_derived_text_writer(record, derived)
    conversion = record.get("neuralese_conversion") or {}
    writes = conversion.get("derived_semantic_text_writes")
    if not isinstance(derived, dict) or not isinstance(writes, list) or len(writes) != 1:
        return None
    attestation = writes[0]
    source = derived.get("source_result")
    turn = source.get("generation_turn") if isinstance(source, dict) else None
    original = derived.get("original_target")
    original_calls = original.get("tool_calls") if isinstance(original, dict) else None
    if not isinstance(attestation, dict) or not isinstance(source, dict) or not isinstance(turn, dict):
        return None
    if (derived.get("schema") != "natlang.derived-equivalent-typed-text-target/1"
            or derived.get("transform_revision") not in ("pure-terminal-eval-finish-to-typed-return/3",
                                                           "pure-terminal-eval-finish-to-typed-return/4")
            or derived.get("derivation_role") != "derived_target_not_original_assistant_action"
            or attestation.get("schema") != "natlang.derived-semantic-text-write/1"
            or attestation.get("role") != "derived-equivalent-pure-terminal-eval-finish-target"
            or attestation.get("derivation_role") != "derived_sft_target"
            or attestation.get("transform_revision") != derived.get("transform_revision")
            or attestation.get("training_admission") is not False
            or attestation.get("learned_vectors") is not False
            or attestation.get("qualification_certificate") is not False
            or attestation.get("runtime_gradient_qualification") is not False
            or attestation.get("original_eval_hidden_states_equivalent") is not False
            or not isinstance(original, dict) or original.get("role") != "assistant"
            or not isinstance(original_calls, list) or len(original_calls) != 1):
        return None
    original_call = original_calls[0]
    original_fn = original_call.get("function") if isinstance(original_call, dict) else None
    if not isinstance(original_fn, dict) or original_fn.get("name") != "eval" or not isinstance(original_fn.get("arguments"), str):
        return None
    try:
        original_args = json.loads(original_fn["arguments"])
    except json.JSONDecodeError:
        return None
    code = original_args.get("code") if isinstance(original_args, dict) else None
    if not isinstance(code, str) or original_args.get("finish") is not True:
        return None
    match = re.fullmatch(
        r'\s*const\s+([A-Za-z_$][\w$]*)\s*(?::\s*Neuralese<string>)?\s*=\s*("(?:\\.|[^"\\])*"|`[^`]*`)\s*;\s*return\s+([A-Za-z_$][\w$]*)\s*;\s*',
        code, re.DOTALL)
    if not match or match.group(1) != match.group(3):
        return None
    if match.group(2).startswith('`'):
        body = match.group(2)[1:-1]
        if any(char in body for char in ('`', '$', '\\')):
            return None
    else:
        try:
            body = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None
    if not isinstance(body, str):
        return None
    target = record.get("target")
    calls = target.get("tool_calls") if isinstance(target, dict) else None
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    fn = call.get("function") if isinstance(call, dict) else None
    try:
        args = json.loads(fn.get("arguments")) if isinstance(fn, dict) and isinstance(fn.get("arguments"), str) else None
    except json.JSONDecodeError:
        return None
    value = args.get("value") if isinstance(args, dict) else None
    write = value.get("$write") if isinstance(value, dict) else None
    if (not isinstance(write, dict) or fn.get("name") != "return_result"
            or args.get("status") != "success" or value != attestation.get("target_write")
            or attestation.get("derived_target_call_id") != call.get("id")
            or attestation.get("derived_target_sha256") != derived.get("derived_target_sha256")
            or attestation.get("target_write_name") != write.get("name")
            or write.get("source") != body or write.get("type") != "Neuralese<string>"
            or not isinstance(write.get("name"), str) or not write["name"].startswith("soft-state:")):
        return None
    derived_call_id = "derived_" + _sha((str(derived.get("original_row_id")) + "\0" +
        str(source.get("writer_node")) + "\0" + str(source.get("body_sha256"))).encode("utf-8"))[:24]
    raw_target = {"role": "assistant", "content": "", "tool_calls": [{"id": derived_call_id,
        "type": "function", "function": {"name": "return_result", "arguments":
            json.dumps({"status": "success", "value": body}, ensure_ascii=False, separators=(",", ":"))}}]}
    if (_sha(_canonical(original).encode("utf-8")) != derived.get("original_target_sha256")
            or _sha(_canonical(raw_target).encode("utf-8")) != derived.get("derived_target_sha256")
            or _sha(_canonical(derived.get("original_messages") or []).encode("utf-8")) != derived.get("original_messages_sha256")
            or _sha(code.encode("utf-8")) != derived.get("original_code_sha256")
            or _sha(body.encode("utf-8")) != source.get("body_sha256")
            or turn.get("invocation_id") != source.get("writer_call_id")
            or turn.get("invocation_id") != (record.get("source_ref") or {}).get("invocation_id")
            or turn.get("trajectory_index") != (record.get("decision") or {}).get("index")
            or turn.get("raw_response_sha256") != (record.get("decision") or {}).get("source_raw_response_sha256")
            or not all(_sha256_hex(turn.get(key)) for key in ("request_sha256", "raw_response_sha256"))
            or source.get("source") != "eval-finish" or source.get("marker_context") != "return-result"
            or source.get("source_kind") != "typed-text-result" or derived.get("source_result_type") != "Neuralese<string>"
            or not _sha256_hex(source.get("source_row_sha256")) or not _sha256_hex(source.get("source_result_row_sha256"))
            or not _sha256_hex(source.get("body_sha256")) or not _sha256_hex(derived.get("derived_target_sha256"))):
        return None
    if (attestation.get("source_row_sha256") != source.get("source_row_sha256")
            or attestation.get("invocation_id") != turn.get("invocation_id")
            or attestation.get("source_writer_call_id") != source.get("writer_call_id")
            or attestation.get("source_writer_node") != source.get("writer_node")
            or attestation.get("source_block_id") != source.get("block_id")
            or attestation.get("source_body_sha256") != source.get("body_sha256")
            or attestation.get("original_messages_sha256") != derived.get("original_messages_sha256")
            or attestation.get("source_generation_turn") != turn):
        return None
    matching = []
    for action in (((record.get("decision") or {}).get("assistant") or {}).get("calls") or []):
        outcome = action.get("outcome") or {}
        for writer in outcome.get("typed_result_writes") or []:
            if (writer.get("schema") == "natlang.typed-result-write/1"
                    and writer.get("source") == "eval-finish"
                    and writer.get("source_kind") == "typed-text-result"
                    and writer.get("invocation_id") == turn.get("invocation_id")
                    and writer.get("writer_call_id") == source.get("writer_call_id")
                    and writer.get("writer_node") == source.get("writer_node")
                    and writer.get("block_id") == source.get("block_id")
                    and writer.get("result_type") == "Neuralese<string>"
                    and writer.get("source_row_sha256") == source.get("source_row_sha256")
                    and writer.get("body_sha256") == source.get("body_sha256")
                    and writer.get("request_sha256") == turn.get("request_sha256")
                    and writer.get("raw_response_sha256") == turn.get("raw_response_sha256")):
                matching.append(writer)
    if len(matching) != 1 or _sha(_canonical(matching[0]).encode("utf-8")) != attestation.get("source_typed_result_receipt_sha256"):
        return None
    if (call.get("id") != derived_call_id
            or _sha(_canonical(args).encode("utf-8")) != attestation.get("derived_action_arguments_sha256")):
        return None
    return {"name": write["name"], "body": body, "body_sha256": source["body_sha256"],
            "block_id": source["block_id"], "source": "derived-equivalent-pure-literal-target"}


def _authenticated_root_derived_text_writer(record, derived):
    """Validate a root-admitted computed text target without treating it as a native write."""
    target = record.get("target") or {}
    calls = target.get("tool_calls") if isinstance(target, dict) else None
    admission_meta = record.get("training_admission") or {}
    admission_path = derived.get("root_admission_path")
    candidate_path = derived.get("conversion_candidate_path")
    if (not isinstance(admission_path, str) or not isinstance(candidate_path, str)
            or not Path(admission_path).is_file() or not Path(candidate_path).is_file()):
        return None
    try:
        admission_bytes = Path(admission_path).read_bytes()
        candidate_bytes = Path(candidate_path).read_bytes()
        admission = json.loads(admission_bytes)
        candidate = json.loads(candidate_bytes)
    except (OSError, json.JSONDecodeError):
        return None
    if (_sha(admission_bytes) != derived.get("root_admission_sha256")
            or _sha(admission_bytes) != admission_meta.get("receipt_sha256")
            or _sha(candidate_bytes) != derived.get("conversion_candidate_sha256")
            or admission.get("schema") != "natlang.root-derived-observed-text-writer-admission/1"
            or admission.get("decision") != "admit-derived-observed-text-writer-body"
            or admission.get("training_admission") is not True
            or admission.get("original_action_admission") is not False
            or admission.get("limits", {}).get("whole_trajectory_admission") is not False
            or admission.get("limits", {}).get("runtime_gradient_qualification") is not False
            or admission.get("limits", {}).get("active_training_inputs_changed") is not False
            or admission_meta.get("kind") != "root-derived-text-writer-body-admission"
            or admission_meta.get("approved") is not True
            or candidate.get("schema") != "natlang.derived-observed-text-writer-candidate/1"
            or candidate.get("proposal_id") != admission.get("proposal_id")
            or candidate.get("exact_body_sha256") != admission.get("exact_body_sha256")
            or candidate.get("derived_supervision", {}).get("target_sha256") != admission.get("derived_target_sha256")
            or derived.get("proposal_id") != admission.get("proposal_id")
            or derived.get("original_native_record_id") != admission.get("source_native_record_id")
            or derived.get("exact_body_sha256") != admission.get("exact_body_sha256")
            or derived.get("target_sha256") != admission.get("derived_target_sha256")
            or derived.get("original_eval_hidden_states_equivalent") is not False
            or derived.get("recurrence_admission") is not False
            or derived.get("runtime_gradient_qualification") is not False
            or record.get("trace_admission", {}).get("admitted") is not False
            or record.get("split") != admission.get("split")
            or admission.get("source_group") not in record.get("source_groups", [])):
        return None
    root = next((parent.parent for parent in Path(admission_path).resolve().parents
                 if parent.name == "runs" and (parent.parent / "training/neuralese_corpora.json").is_file()), None)
    if root is None:
        return None
    for rel, pin in (admission.get("input_pins") or {}).items():
        path = (root / rel).resolve()
        if (not path.is_relative_to(root) or not path.is_file() or not isinstance(pin, dict)
                or _sha(path.read_bytes()) != pin.get("sha256")
                or path.stat().st_size != pin.get("bytes")):
            return None
    conversion = candidate.get("conversion_provenance") or {}
    converter_path = conversion.get("converter_path")
    verifier_path = conversion.get("closure_verifier_path")
    closure = conversion.get("verified_code_closure") or {}
    manifest_path = closure.get("manifest_path")
    if not all(isinstance(x, str) and Path(x).is_file() for x in (converter_path, verifier_path, manifest_path)):
        return None
    if (_sha(Path(converter_path).read_bytes()) != conversion.get("converter_sha256")
            or _sha(Path(verifier_path).read_bytes()) != conversion.get("closure_verifier_sha256")
            or _sha(Path(manifest_path).read_bytes()) != closure.get("manifest_sha256")):
        return None
    try:
        manifest = json.loads(Path(manifest_path).read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    entries = manifest.get("closure_files")
    if not isinstance(entries, list) or not entries:
        return None
    verified = []
    manifest_dir = Path(manifest_path).resolve().parent
    for entry in entries:
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(rel, str):
            return None
        path = (manifest_dir / rel).resolve()
        if not path.is_relative_to(manifest_dir) or not path.is_file():
            return None
        contents = path.read_bytes()
        if (_sha(contents) != entry.get("sha256")
                or (entry.get("bytes") is not None and len(contents) != entry.get("bytes"))):
            return None
        verified.append({"path": rel, "sha256": _sha(contents), "bytes": len(contents)})
    if (_sha(json.dumps(verified, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            != closure.get("closure_sha256") or len(verified) != closure.get("closure_files_verified")):
        return None
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    fn = call.get("function") if isinstance(call, dict) else None
    try:
        args = json.loads(fn.get("arguments")) if isinstance(fn, dict) and isinstance(fn.get("arguments"), str) else None
    except json.JSONDecodeError:
        return None
    value = args.get("value") if isinstance(args, dict) else None
    body = value if isinstance(value, str) else None
    if (fn.get("name") != "return_result" or args.get("status") != "success"
            or body is None or _sha(body.encode("utf-8")) != admission.get("exact_body_sha256")
            or _sha(json.dumps(target, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                != admission.get("derived_target_sha256")
            or _sha(json.dumps(derived.get("original_target"), ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")) != derived.get("original_target_sha256")
            or _sha(json.dumps(record.get("messages") or [], ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")) != derived.get("original_messages_sha256")):
        return None
    return {"body": body, "body_sha256": admission["exact_body_sha256"],
            "source": "root-admitted-derived-text-writer-body", "native_runtime_write": False}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tokenizer_fingerprint(tokenizer) -> str:
    """Bind pretokenized text to exact token coordinates and chat formatting."""
    return _sha(_canonical({"vocab": tokenizer.get_vocab(),
                            "chat_template": tokenizer.chat_template,
                            "special_tokens": sorted(tokenizer.all_special_tokens),
                            "tokenizer_backend": json.loads(tokenizer.backend_tokenizer.to_str())}).encode("utf-8"))


def _native_gold_render(tokenizer, turns, tools):
    """Use the serving renderer; content markers remain ordinary content tokens."""
    from ..serve.chat import render_messages, render_with_empty_thought, split_escaped
    def template(turns, schemas):
        return render_with_empty_thought(lambda m: tokenizer.apply_chat_template(
            m, tools=schemas or None, tokenize=False, add_generation_prompt=False), turns)
    rendered = render_messages(turns, tools, template,
                               specials=(*tokenizer.all_special_tokens, "<|neuralese|>", "<|/neuralese|>"))
    if rendered.blocks:
        raise ValueError("ordinary gold text contains unresolved neuralese blocks")
    text, ids = [], []
    for segment in rendered.segments:
        for run, escaped in split_escaped(segment, rendered.escape_nonce):
            text.append(run)
            ids.extend(tokenizer(run, add_special_tokens=False,
                                 split_special_tokens=escaped)["input_ids"])
    return "".join(text), ids


def _message_neuralese_ids(value):
    if isinstance(value, dict):
        if value.get("type") == "neuralese" and isinstance(value.get("id"), str):
            yield value["id"]
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    child = json.loads(child)
                except json.JSONDecodeError:
                    pass
            yield from _message_neuralese_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _message_neuralese_ids(child)


def _message_soft_reads(value):
    """Yield explicit soft-state read parts without interpreting free text."""
    if isinstance(value, dict):
        if value.get("type") == "read" and isinstance(value.get("name"), str):
            yield value
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    child = json.loads(child)
                except json.JSONDecodeError:
                    pass
            yield from _message_soft_reads(child)
    elif isinstance(value, list):
        for child in value:
            yield from _message_soft_reads(child)


def _soft_writer_sources(records, source_hashes, *, preview_only=False):
    """Index only admitted or explicitly held-preview successful writes as context bodies.

    These bodies are sourced from the exact `$write.source` in an approved
    producer target. They hydrate matching reader context; they do not create
    extra target rows or independent gold labels.
    """
    records = list(records)
    sources = {}

    def add_source(block_id, attestation):
        same_writer = [item for item in sources.setdefault(block_id, [])
                       if item.get("writer_record_id") == attestation.get("writer_record_id")
                       and item.get("body_sha256") == attestation.get("body_sha256")
                       and item.get("body") == attestation.get("body")]
        occurrence = {key: attestation[key] for key in
                      ("writer_write_node", "result_path", "write_name", "source_kind") if key in attestation}
        if len(same_writer) == 1:
            same_writer[0].setdefault("write_occurrences", []).append(occurrence)
        else:
            attestation["write_occurrences"] = [occurrence]
            sources[block_id].append(attestation)

    def add_write(write, record):
        if (isinstance(write, dict) and isinstance(write.get("name"), str)
                and write["name"].startswith("soft-state:")
                and write.get("type") == "Neuralese<string>"
                and isinstance(write.get("source"), str)):
            block_id = write.get("block_id") or write["name"].split(":", 1)[1]
            if not isinstance(block_id, str) or not block_id.startswith("nz1_"):
                return
            attestation = {"body": write["source"], "writer_record_id": record.get("id"),
                           "writer_record_sha256": source_hashes.get(record.get("id")),
                           "writer_source_row_sha256": ((record.get("source_ref") or {}).get("source_row_sha256")),
                           "writer_split": record.get("split"),
                           "writer_source_groups": sorted(set(g for g in (record.get("source_groups") or [])
                                                                if isinstance(g, str) and g)),
                           "write_name": write["name"], "body_sha256": _sha(write["source"].encode("utf-8"))}
            add_source(block_id, attestation)

    def visit(value, record):
        if isinstance(value, dict):
            write = value.get("$write")
            add_write(write, record)
            for child in value.values():
                visit(child, record)
        elif isinstance(value, list):
            for child in value:
                visit(child, record)

    for record in records:
        decision = record.get("decision") or {}
        if record.get("derived_target") is not None:
            # A transformed pure-literal target is an SFT view, not a runtime
            # writer that can close a recurrence edge into another record.
            continue
        admitted = (not preview_only
                    and record.get("training_admission", {}).get("approved") is True
                    and decision.get("training_approved") is True
                    and decision.get("failed_action") is False)
        held_preview = (preview_only is True
                        and record.get("review_disposition") == "held_for_root_review"
                        and record.get("training_admission", {}).get("approved") is not True
                        and decision.get("failed_action") is False)
        if not (admitted or held_preview):
            continue
        for call in (record.get("target") or {}).get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                continue
            visit(args, record)
            sidecar = call.get("neuralese_code")
            if sidecar is not None:
                from ..train.inline_instructions import validate_inline_instruction_code
                checked = validate_inline_instruction_code(raw, sidecar)
                if checked.valid:
                    for write in checked.value.writes:
                        add_write({"name": write.name, "type": write.type, "source": write.source}, record)

        source_ref = record.get("source_ref") or {}
        for call in ((decision.get("assistant") or {}).get("calls") or []):
            for receipt in ((call.get("outcome") or {}).get("typed_result_writes") or []):
                if (not isinstance(receipt, dict)
                        or receipt.get("schema") != "natlang.typed-result-write/1"
                        or receipt.get("invocation_id") != source_ref.get("invocation_id")
                        or receipt.get("source_row_sha256") != source_ref.get("source_row_sha256")
                        or receipt.get("source_kind") not in {"typed-text-result", "typed-text-result-field"}
                        or receipt.get("result_type") != "Neuralese<string>"
                        or not isinstance(receipt.get("writer_node"), str)
                        or not isinstance(receipt.get("block_id"), str)
                        or not receipt["block_id"].startswith("nz1_")
                        or not isinstance(receipt.get("body_source"), str)
                        or not isinstance(receipt.get("body_sha256"), str)
                        or _sha(receipt["body_source"].encode("utf-8")) != receipt["body_sha256"]
                        or not isinstance(receipt.get("result_path"), list)
                        or not receipt["result_path"] or receipt["result_path"][0] != "return"):
                    continue
                add_source(receipt["block_id"], {"body": receipt["body_source"],
                    "writer_record_id": record.get("id"), "writer_record_sha256": source_hashes.get(record.get("id")),
                    "writer_source_row_sha256": source_ref.get("source_row_sha256"),
                    "writer_split": record.get("split"),
                    "writer_source_groups": sorted(set(g for g in (record.get("source_groups") or [])
                                                         if isinstance(g, str) and g)),
                    "write_name": (_direct_result_event_name(receipt["block_id"], source_ref.get("trajectory_id"),
                                                              receipt["writer_call_id"], receipt["writer_node"])
                                   if receipt.get("source_kind") == "typed-text-result"
                                   and receipt.get("source") == "return_result"
                                   and receipt.get("body_source_basis") == "exact-raw-model-result-string"
                                   else "soft-state:" + receipt["block_id"]),
                    "body_sha256": receipt["body_sha256"],
                    "writer_invocation_id": receipt["invocation_id"], "writer_write_node": receipt["writer_node"],
                    "result_path": receipt["result_path"], "source_kind": "runtime-typed-result-write"})

    rows_by_invocation = {}
    for record in records:
        source_ref = record.get("source_ref") or {}
        rows_by_invocation.setdefault((source_ref.get("trajectory_id"), source_ref.get("invocation_id")), []).append(record)
    for reader in records:
        reader_ref = reader.get("source_ref") or {}
        for context in reader_ref.get("provider_expanded_read_contexts") or []:
            if not isinstance(context, dict) or context.get("origin") != "same-run-producer":
                continue
            writer_event, block = context.get("producer_write"), context.get("block")
            if not isinstance(writer_event, dict) or not isinstance(block, dict):
                continue
            writer_call, writer_node = writer_event.get("call_id"), writer_event.get("node")
            block_id, body, body_sha = block.get("id"), block.get("body"), block.get("body_sha256")
            if (not isinstance(writer_call, str) or not isinstance(writer_node, str)
                    or not isinstance(block_id, str) or not isinstance(body, str)
                    or not isinstance(body_sha, str) or _sha(body.encode("utf-8")) != body_sha
                    or writer_event.get("block") != block_id or writer_event.get("text_body_sha256") != body_sha):
                continue
            for writer_record in rows_by_invocation.get((reader_ref.get("trajectory_id"), writer_call), []):
                writer_ref = writer_record.get("source_ref") or {}
                if writer_ref.get("source_row_sha256") != reader_ref.get("source_row_sha256"):
                    continue
                receipts = [receipt for call in ((((writer_record.get("decision") or {}).get("assistant") or {}).get("calls")) or [])
                            for receipt in ((call.get("outcome") or {}).get("typed_result_writes") or [])
                            if isinstance(receipt, dict) and receipt.get("writer_node") == writer_node
                            and receipt.get("block_id") == block_id and receipt.get("body_sha256") == body_sha]
                if len(receipts) != 1 or receipts[0].get("body_source") is not None:
                    continue
                receipt = receipts[0]
                add_source(block_id, {"body": body, "writer_record_id": writer_record.get("id"),
                    "writer_record_sha256": source_hashes.get(writer_record.get("id")),
                    "writer_source_row_sha256": writer_ref.get("source_row_sha256"),
                    "writer_split": writer_record.get("split"),
                    "writer_source_groups": sorted(set(g for g in (writer_record.get("source_groups") or [])
                                                         if isinstance(g, str) and g)),
                    "write_name": "soft-state:" + block_id, "body_sha256": body_sha,
                    "writer_invocation_id": writer_call, "writer_write_node": writer_node,
                    "result_path": receipt.get("result_path"),
                    "source_kind": "same-run-provider-expanded-writer-context"})
    return sources


def authenticated_context_writer_sources(records, *, preview_only=False):
    """Build the shared source-hash and approved-writer index for context rendering.

    Recurrence training uses the same writer evidence as ordinary gold-text
    rendering so a typed context can become a differentiable reader edge only
    when one exact same-split, same-source writer is available.
    """
    record_rows = list(records)
    source_hashes = {
        record.get("id", ""): (record.get("_source_record_sha256") or
            _sha(_canonical(dict(record)).encode("utf-8")))
        for record in record_rows
    }
    return source_hashes, _soft_writer_sources(record_rows, source_hashes, preview_only=preview_only)


def _attested_neuralese_message_bodies(record, writer_sources=None, *, split, source_groups,
                                       authenticated_context_bodies=None):
    """Map exact message blocks to hash-bound writer or creation source text."""
    messages = record.get("messages") or []
    message_block_ids = list(_message_neuralese_ids(messages))
    block_ids = sorted(set(message_block_ids))
    if not block_ids:
        return {}, []
    site = (((record.get("source_ref") or {}).get("inline_instruction_site") or {}).get("site") or {})
    body_id, body = site.get("soft_body_id"), site.get("raw_body_source")
    bodies, attestations = {}, []
    for block_id in block_ids:
        reader_source_row = ((record.get("source_ref") or {}).get("source_row_sha256"))
        eligible_writers = [writer for writer in (writer_sources or {}).get(block_id, [])
                            if writer.get("writer_split") == split
                            and (set(writer.get("writer_source_groups") or []) & set(source_groups)
                                 or (isinstance(reader_source_row, str) and reader_source_row
                                     and writer.get("writer_source_row_sha256") == reader_source_row))]
        writer = eligible_writers[0] if len(eligible_writers) == 1 else None
        if (block_id == body_id and message_block_ids.count(block_id) == 1
                and isinstance(body, str) and body.strip()
                and "<|neuralese|>" not in body and "<|/neuralese|>" not in body
                and site.get("raw_body_source_sha256") == _sha(body.encode("utf-8"))):
            attestation = {"body": body, "writer_record_id": record.get("id"),
                           "writer_record_sha256": record.get("_source_record_sha256"),
                           "writer_source_row_sha256": ((record.get("source_ref") or {}).get("source_row_sha256")),
                           "write_name": "inline_instruction_site", "body_sha256": _sha(body.encode("utf-8")),
                           "source_kind": "hash_bound_creation_body"}
        elif writer is not None:
            attestation = {**writer, "source_kind": "approved_writer_target_source"}
        elif isinstance(authenticated_context_bodies, dict) and block_id in authenticated_context_bodies:
            context = authenticated_context_bodies[block_id]
            attestation = {key: value for key, value in context.items() if key != "body"} | {
                "body": context["body"], "source_kind": "provider_expanded_context_only_input"}
        else:
            raise ValueError(f"message neuralese body has no unique same-split, same-source hash-bound source: {block_id}")
        bodies[block_id] = f"<|neuralese|>{attestation['body']}<|/neuralese|>"
        attestations.append({key: value for key, value in attestation.items() if key != "body"}
                            | {"block_id": block_id})
    return bodies, attestations


def _attested_provider_expanded_reads(record, writer_sources=None, *, split, source_groups):
    """Attest exact crisp stand-in reads against their authenticated same-run writer.

    The reader stays a typed ``read`` in native/R data, while ordinary gold text
    renders its exact source string. This receipt makes the grounding explicit
    without claiming the body was a learned vector or a new target.
    """
    conversion = record.get("neuralese_conversion") or {}
    metadata = [item for item in conversion.get("external_context_inputs", [])
                if isinstance(item, dict)
                and item.get("schema") == "natlang.external-context-input/1"
                and ((item.get("origin") == "same-run-producer"
                      and item.get("learner_representation") in {
                    "typed-read-linked-to-existing-writer",
                    "typed-read-from-authenticated-runtime-writer-event-context-only"})
                     or (item.get("origin") == "runtime-definition-context-only"
                         and item.get("learner_representation") == "runtime-definition-context-only"))]
    if not metadata:
        return []
    reads = list(_message_soft_reads(record.get("messages") or []))
    attestations = []
    reader_source_row = ((record.get("source_ref") or {}).get("source_row_sha256"))
    reader_invocation = ((record.get("source_ref") or {}).get("invocation_id"))
    for item in metadata:
        block_id = item.get("block_id")
        body_sha256 = item.get("body_sha256")
        if item.get("origin") == "runtime-definition-context-only":
            refs = [receipt for receipt in ((record.get("source_ref") or {}).get(
                "provider_expanded_read_contexts") or [])
                    if isinstance(receipt, dict)
                    and receipt.get("schema") == "natlang.provider-expanded-read-context/1"
                    and receipt.get("origin") == "configured-function-definition"
                    and (receipt.get("block") or {}).get("id") == block_id]
            if len(refs) != 1:
                raise ValueError("runtime-definition context lacks one exact configured-read receipt")
            receipt = refs[0]
            block, read, turn = (receipt.get("block") or {}, receipt.get("block_read") or {},
                                 receipt.get("model_turn") or {})
            readout, definition = receipt.get("readout") or {}, receipt.get("definition") or {}
            body = block.get("body")
            pairs = [(item.get("type"), block.get("type")),
                     (item.get("body_sha256"), block.get("body_sha256")),
                     (item.get("invocation_id"), receipt.get("invocation_id")),
                     (item.get("source_row_sha256"), receipt.get("source_row_sha256")),
                     (item.get("trace_sha256"), receipt.get("trace_sha256")),
                     (item.get("transport_provenance_sha256"), receipt.get("transport_provenance_sha256")),
                     (item.get("raw_request_sha256"), receipt.get("raw_request_sha256")),
                     (item.get("rendered_request_sha256"), receipt.get("rendered_request_sha256")),
                     (item.get("source_request_sha256"), receipt.get("source_request_sha256")),
                     (item.get("source_response_sha256"), receipt.get("source_response_sha256")),
                     (item.get("read_node"), read.get("node")),
                     (item.get("model_turn_node"), turn.get("node")),
                     (item.get("runtime_definition_id"), definition.get("id"))]
            if (any(left != right for left, right in pairs)
                    or item.get("learner_representation") != "runtime-definition-context-only"
                    or item.get("writer_target_selected") is not False
                    or item.get("learned_vectors") is not False
                    or item.get("qualification_certificate") is not False
                    or item.get("training_admission") is not False
                    or item.get("invocation_id") != reader_invocation
                    or item.get("source_row_sha256") != reader_source_row
                    or not isinstance(item.get("type"), str) or not item["type"].startswith("Neuralese<")
                    or not isinstance(body, str) or _sha(body.encode("utf-8")) != body_sha256
                    or not isinstance(item.get("runtime_manifest_sha256"), str)
                    or not isinstance(item.get("runtime_definition_source_sha256"), str)
                    or not isinstance(item.get("runtime_definition_source"), str)
                    or readout.get("read_body_id") != block_id or readout.get("read_source_sha256") != body_sha256
                    or any(readout.get(flag) is not False for flag in
                           ("learned_vectors", "qualification_certificate", "training_admission"))
                    or receipt.get("producer_write") is not None
                    or read.get("kind") != "block_read" or read.get("block") != block_id
                    or read.get("call_id") != reader_invocation or turn.get("kind") != "model_turn"
                    or turn.get("call_id") != reader_invocation
                    or not any(isinstance(edge, dict) and edge.get("node") == read.get("node")
                               and edge.get("block") == block_id for edge in turn.get("inputs", []))
                    or typed_neuralese_ref_count(record.get("messages") or [], block_id)
                       != item.get("context_occurrences")):
                raise ValueError("runtime-definition context binding does not match its exact read and model turn")
            attestations.append({key: value for key, value in item.items() if key != "body"} | {
                "block_id": block_id, "reader_record_id": record.get("id"), "body": body,
                "source_kind": "provider-expanded-runtime-definition-context-only"})
            continue
        if (item.get("type") != "Neuralese<string>" or not isinstance(block_id, str)
                or not block_id.startswith("nz1_") or not isinstance(body_sha256, str)
                or item.get("learned_vectors") is not False
                or item.get("qualification_certificate") is not False
                or item.get("training_admission") is not False
                or item.get("invocation_id") != reader_invocation
                or item.get("source_row_sha256") != reader_source_row
                or not isinstance(item.get("read_node"), str)
                or not isinstance(item.get("model_turn_node"), str)
                or not isinstance(item.get("producer_write_node"), str)
                or not isinstance(item.get("transport_provenance_sha256"), str)):
            raise ValueError("provider-expanded read context metadata is incomplete")
        name = item.get("target_write_name") or "soft-state:" + block_id
        context_refs = [receipt for receipt in ((record.get("source_ref") or {}).get(
            "provider_expanded_read_contexts") or [])
                        if isinstance(receipt, dict)
                        and receipt.get("schema") == "natlang.provider-expanded-read-context/2"
                        and receipt.get("origin") == "same-run-producer"
                        and (receipt.get("block") or {}).get("id") == block_id]
        if len(context_refs) > 1 or (item.get("target_write_name") is not None and len(context_refs) != 1):
            raise ValueError("provider-expanded context lacks one exact source receipt")
        producer = context_refs[0].get("producer_write") or {} if context_refs else {}
        direct_return = (producer.get("source") == "return_result"
                         and producer.get("source_kind") == "typed-text-result"
                         and producer.get("result_type") == "Neuralese<string>"
                         and isinstance(producer.get("call_id"), str)
                         and isinstance(producer.get("node"), str))
        expected_name = (_direct_result_event_name(block_id, (record.get("source_ref") or {}).get("trajectory_id"),
                                                   producer["call_id"], producer["node"])
                         if direct_return and item.get("target_write_name") is not None
                         else "soft-state:" + block_id)
        if name != expected_name:
            raise ValueError("provider-expanded read name does not match its exact producer event")
        matches = [part for part in reads if part.get("name") == name]
        if matches:
            if any(not isinstance(part.get("source"), str) for part in matches):
                raise ValueError("provider-expanded read does not have an exact typed source")
            body = matches[0]["source"]
            if any(part["source"] != body for part in matches) or _sha(body.encode("utf-8")) != body_sha256:
                raise ValueError("provider-expanded read source digest mismatch")
        else:
            # Some native turns carry the exact provider-expanded typed input
            # directly as a Neuralese message block rather than a crisp read
            # object. Hydrate that reference only from the existing exact
            # context-only graph receipt; do not infer a writer target or edge.
            typed_occurrences = typed_neuralese_ref_count(record.get("messages") or [], block_id)
            expected_occurrences = item.get("context_occurrences")
            if (typed_occurrences < 1 or type(expected_occurrences) is not int
                    or typed_occurrences != expected_occurrences
                    or item.get("writer_target_selected") is not False):
                raise ValueError("provider-expanded typed message reference lacks an exact context-only binding")
            context_receipts = [receipt for receipt in ((record.get("source_ref") or {}).get(
                "provider_expanded_read_contexts") or [])
                                if isinstance(receipt, dict)
                                and receipt.get("schema") == "natlang.provider-expanded-read-context/2"
                                and receipt.get("origin") == "same-run-producer"
                                and (receipt.get("block") or {}).get("id") == block_id]
            if len(context_receipts) != 1:
                raise ValueError("provider-expanded typed message reference lacks one exact source receipt")
            source_block = context_receipts[0].get("block") or {}
            body = source_block.get("body")
            if (source_block.get("type") != item.get("type") or not isinstance(body, str)
                    or source_block.get("body_sha256") != body_sha256
                    or _sha(body.encode("utf-8")) != body_sha256):
                raise ValueError("provider-expanded typed message body digest mismatch")
        eligible = [writer for writer in (writer_sources or {}).get(block_id, [])
                    if writer.get("writer_split") == split
                    and (set(writer.get("writer_source_groups") or []) & set(source_groups)
                         or (isinstance(reader_source_row, str) and reader_source_row
                             and writer.get("writer_source_row_sha256") == reader_source_row))
                    and ((writer.get("writer_write_node") is None and
                          not any(isinstance(item, dict) and item.get("writer_write_node") is not None
                                  for item in writer.get("write_occurrences", []))) or
                         writer.get("writer_write_node") == item.get("producer_write_node") or
                         any(isinstance(occurrence, dict) and
                             occurrence.get("writer_write_node") == item.get("producer_write_node")
                             for occurrence in writer.get("write_occurrences", [])))
                    and writer.get("body_sha256") == body_sha256
                    and writer.get("body") == body]
        if len(eligible) > 1:
            coalesced = _coalesce_text_writer_event_aliases(
                eligible, block_id=block_id, block_type=item.get("type"), body=body,
                body_sha256=body_sha256,
                writer_call_id=(producer.get("call_id") if context_refs else item.get("producer_call_id")),
                writer_write_node=item.get("producer_write_node"), writer_name=name,
                source_row_sha256=reader_source_row, split=split, source_groups=source_groups)
            if coalesced is not None:
                eligible = [coalesced]
        if len(eligible) == 1:
            writer = eligible[0]
            attestations.append({key: value for key, value in writer.items() if key != "body"} | {
                "block_id": block_id, "reader_record_id": record.get("id"),
                "read_node": item["read_node"], "model_turn_node": item["model_turn_node"],
                "producer_write_node": item["producer_write_node"],
                "transport_provenance_sha256": item["transport_provenance_sha256"],
                "writer_target_selected": item.get("writer_target_selected"), "body": body,
                "source_kind": "provider-expanded-same-run-read"})
            continue
        if eligible:
            raise ValueError("provider-expanded read has ambiguous same-split, same-source writers")
        if item.get("writer_target_selected") is not False:
            raise ValueError("provider-expanded read has no unique same-split, same-source writer")
        source_ref = record.get("source_ref") or {}
        receipts = [receipt for receipt in source_ref.get("provider_expanded_read_contexts", [])
                    if isinstance(receipt, dict) and receipt.get("schema") == "natlang.provider-expanded-read-context/2"
                    and receipt.get("origin") == "same-run-producer" and receipt.get("block", {}).get("id") == block_id]
        if len(receipts) != 1:
            raise ValueError("context-only read lacks one hash-bound provider receipt")
        receipt = receipts[0]
        block = receipt.get("block") or {}
        write = receipt.get("producer_write") or {}
        read = receipt.get("block_read") or {}
        turn = receipt.get("model_turn") or {}
        source_binding_fields = (receipt.get("source_action_target_sha256"),
                                 receipt.get("source_trajectory_index"),
                                 receipt.get("source_request_sha256"),
                                 receipt.get("source_response_sha256"))
        source_binding_present = any(value is not None for value in source_binding_fields)
        writer_source_class = receipt.get("writer_source_class")
        writer_source_valid = (
            writer_source_class in (None, "modern-typed-text-result")
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
        ) or (
            writer_source_class == "legacy-text-marker-standin-eval-code"
            and write.get("emulation_version") == "text-marker-standin/2"
            and write.get("marker_context") == "eval-code"
            and write.get("learned_vectors") is False
        ) or (
            writer_source_class == "legacy-text-marker-standin-return-result"
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
            and write.get("source") == "return_result"
            and write.get("marker_context") == "return-result"
            and isinstance(receipt.get("writer_witness"), dict)
            and receipt["writer_witness"].get("kind") == "raw-return-result-value-equals-expanded-body"
            and receipt["writer_witness"].get("source") == "return_result"
            and receipt["writer_witness"].get("host_result_call_id") == write.get("call_id")
            and receipt["writer_witness"].get("host_result_type") == block.get("type")
            and _sha256_hex(receipt["writer_witness"].get("host_result_value_sha256"))
            and _sha256_hex(receipt["writer_witness"].get("raw_response_sha256"))
        ) or (
            writer_source_class == "legacy-text-marker-standin-eval-finish"
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
            and write.get("source") == "eval-finish"
            and write.get("marker_context") == "return-result"
            and isinstance(receipt.get("writer_witness"), dict)
            and receipt["writer_witness"].get("kind") == "completed-eval-finish-host-reference"
            and receipt["writer_witness"].get("source") == "eval-finish"
            and receipt["writer_witness"].get("host_result_call_id") == write.get("call_id")
            and receipt["writer_witness"].get("host_result_type") == block.get("type")
            and _sha256_hex(receipt["writer_witness"].get("host_result_value_sha256"))
        ) or (
            writer_source_class == "legacy-text-marker-standin-eval-return"
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
            and write.get("source") == "eval-return"
            and write.get("marker_context") == "return-result"
            and isinstance(receipt.get("writer_witness"), dict)
            and receipt["writer_witness"].get("kind") == "completed-eval-return-host-reference"
            and receipt["writer_witness"].get("source") == "eval-return"
            and receipt["writer_witness"].get("host_result_call_id") == write.get("call_id")
            and receipt["writer_witness"].get("host_result_type") == block.get("type")
            and receipt["writer_witness"].get("completion_status") == "done"
            and receipt["writer_witness"].get("completion_source") == "execution_graph"
            and receipt["writer_witness"].get("completion_detail") == f"\uE000{block_id}\uE001"
            and _sha256_hex(receipt["writer_witness"].get("host_result_value_sha256"))
        )
        repeated_pairs = receipt.get("additional_read_turn_pairs") or []
        required_hashes = (body_sha256, item.get("source_row_sha256"), item.get("trace_sha256"),
                           item.get("transport_provenance_sha256"), item.get("raw_request_sha256"),
                           item.get("rendered_request_sha256"), receipt.get("source_row_sha256"),
                           receipt.get("trace_sha256"), receipt.get("transport_provenance_sha256"),
                           receipt.get("raw_request_sha256"), receipt.get("rendered_request_sha256"),
                           block.get("body_sha256"), write.get("text_body_sha256"))
        if source_binding_present:
            required_hashes += (receipt.get("source_action_target_sha256"), receipt.get("source_request_sha256"),
                                receipt.get("source_response_sha256"))
        valid_repeated_pairs = isinstance(repeated_pairs, list) and all(
            isinstance(pair, dict) and isinstance(pair.get("block_read"), dict)
            and isinstance(pair.get("model_turn"), dict)
            and pair["block_read"].get("kind") == "block_read"
            and pair["model_turn"].get("kind") == "model_turn"
            and _nonempty_string(pair["block_read"].get("node"))
            and _nonempty_string(pair["model_turn"].get("node"))
            and pair["block_read"].get("turn") == pair["model_turn"].get("node")
            and pair["block_read"].get("call_id") == reader_invocation
            and pair["block_read"].get("block") == block_id
            and pair["model_turn"].get("call_id") == reader_invocation
            and any(isinstance(inp, dict) and inp.get("node") == write.get("node")
                    and inp.get("block") == block_id for inp in pair["block_read"].get("inputs", []))
            and any(isinstance(inp, dict) and inp.get("node") == pair["block_read"].get("node")
                    and inp.get("port") == "read" and inp.get("block") == block_id
                    for inp in pair["model_turn"].get("inputs", []))
            for pair in repeated_pairs)
        if (receipt.get("invocation_id") != reader_invocation
                or receipt.get("source_row_sha256") != reader_source_row
                or item.get("trace_sha256") != ((record.get("provenance") or {}).get("trace_sha256"))
                or receipt.get("trace_sha256") != ((record.get("provenance") or {}).get("trace_sha256"))
                or receipt.get("writer_target_selected") is not False
                or receipt.get("learned_vectors") is not False
                or receipt.get("qualification_certificate") is not False
                or receipt.get("training_admission") is not False
                or receipt.get("parent_invocation_id") != item.get("parent_invocation_id")
                or item.get("writer_source_class") != writer_source_class
                or item.get("writer_witness") != receipt.get("writer_witness")
                or block.get("type") != item.get("type") or block.get("body") != body
                or block.get("body_sha256") != body_sha256 or _sha(body.encode("utf-8")) != body_sha256
                or read.get("kind") != "block_read" or turn.get("kind") != "model_turn"
                or read.get("turn") != turn.get("node")
                or not _nonempty_string(write.get("node")) or not _nonempty_string(write.get("call_id"))
                or not _nonempty_string(read.get("node")) or not _nonempty_string(turn.get("node"))
                or not _nonempty_string(item.get("producer_write_node"))
                or not _nonempty_string(item.get("producer_call_id"))
                or not all(_sha256_hex(value) for value in required_hashes)
                or read.get("node") != item.get("read_node") or read.get("call_id") != reader_invocation
                or read.get("block") != block_id or turn.get("node") != item.get("model_turn_node")
                or turn.get("call_id") != reader_invocation
                or not any(inp.get("node") == read.get("node") and inp.get("port") == "read"
                           and inp.get("block") == block_id for inp in turn.get("inputs", []))
                or write.get("kind") != "block_write" or write.get("block") != block_id
                or write.get("call_id") != item.get("producer_call_id")
                or write.get("node") != item.get("producer_write_node")
                or write.get("result_type") != block.get("type") or write.get("truncated") is not False
                or not writer_source_valid
                or write.get("text_body_sha256") != body_sha256
                or not any(inp.get("node") == write.get("node") and inp.get("block") == block_id
                           for inp in read.get("inputs", []))
                or not valid_repeated_pairs
                or item.get("additional_read_nodes", []) != [pair.get("block_read", {}).get("node") for pair in repeated_pairs]
                or item.get("additional_model_turn_nodes", []) != [pair.get("model_turn", {}).get("node") for pair in repeated_pairs]
                or any(pair.get("block_read", {}).get("call_id") != reader_invocation
                       or pair.get("block_read", {}).get("block") != block_id
                       or pair.get("model_turn", {}).get("call_id") != reader_invocation
                       or not any(inp.get("node") == write.get("node") and inp.get("block") == block_id
                                  for inp in pair.get("block_read", {}).get("inputs", []))
                       or not any(inp.get("node") == pair.get("block_read", {}).get("node")
                                  and inp.get("port") == "read" and inp.get("block") == block_id
                                  for inp in pair.get("model_turn", {}).get("inputs", []))
                       for pair in repeated_pairs)
                or not isinstance(receipt.get("transport_provenance_sha256"), str)
                or not isinstance(receipt.get("raw_request_sha256"), str)
                or not isinstance(receipt.get("rendered_request_sha256"), str)
                or item.get("raw_request_sha256") != receipt.get("raw_request_sha256")
                or item.get("rendered_request_sha256") != receipt.get("rendered_request_sha256")
                or item.get("transport_provenance_sha256") != receipt.get("transport_provenance_sha256")
                or item.get("parent_invocation_id") != receipt.get("parent_invocation_id")):
            raise ValueError("context-only provider read receipt does not authenticate exact body and graph")
        if source_binding_present:
            # A separately admitted derived text writer keeps provider reads
            # bound to the original sampled action. Its converted target is a
            # new supervision object and is not the action that made the read.
            derived = record.get("derived_target") or {}
            selected_action_target = (derived.get("original_target")
                                      if derived.get("schema") ==
                                      "natlang.root-admitted-derived-text-writer-target/1"
                                      else record.get("target"))
            protected_target_sha = _sha(_canonical(selected_action_target).encode("utf-8"))
            target_adapter = receipt.get("target_binding_adapter")
            adapter_valid = (_protected_target_sidecar_equivalence(record, receipt, item)
                             if target_adapter is not None else False)
            if (type(receipt.get("source_trajectory_index")) is not int
                    or receipt.get("source_trajectory_index") != (record.get("decision") or {}).get("index")
                    or (receipt.get("source_action_target_sha256") != protected_target_sha and not adapter_valid)
                    or (target_adapter is not None and not adapter_valid)
                    or item.get("source_action_target_sha256") != receipt.get("source_action_target_sha256")
                    or item.get("source_trajectory_index") != receipt.get("source_trajectory_index")
                    or item.get("source_request_sha256") != receipt.get("source_request_sha256")
                    or item.get("source_response_sha256") != receipt.get("source_response_sha256")):
                raise ValueError("provider-expanded context selected-action binding mismatch")
        expected_read_name = item.get("target_write_name") or "soft-state:" + block_id
        actual_context_occurrences = (context_ref_count(record.get("messages") or [], block_id,
                                                        expected_read_name)
                                      if matches else
                                      typed_neuralese_ref_count(record.get("messages") or [], block_id))
        receipt_context_occurrences = receipt.get("context_occurrences")
        metadata_context_occurrences = item.get("context_occurrences")
        if type(receipt_context_occurrences) is not int or receipt_context_occurrences < 1:
            raise ValueError("provider-expanded context occurrence count is not an exact positive integer")
        if metadata_context_occurrences is None:
            # Older authenticated schema/2 receipts predate this converter
            # metadata field. Recover only the unambiguous single-reference
            # case from the exact typed messages and provider receipt.
            if receipt_context_occurrences != 1 or actual_context_occurrences != 1:
                raise ValueError("legacy provider context occurrence count is not a single exact typed reference")
        elif (type(metadata_context_occurrences) is not int or
              metadata_context_occurrences != receipt_context_occurrences or
              actual_context_occurrences != receipt_context_occurrences):
            raise ValueError("provider-expanded context occurrence count does not match exact typed references")
        attestations.append({"block_id": block_id, "reader_record_id": record.get("id"),
            "writer_record_id": None, "writer_source_row_sha256": reader_source_row,
            "writer_invocation_id": write.get("call_id"), "writer_write_node": write.get("node"),
            "writer_target_selected": False, "read_node": item["read_node"],
            "model_turn_node": item["model_turn_node"], "producer_write_node": item["producer_write_node"],
            "transport_provenance_sha256": item["transport_provenance_sha256"],
            "raw_request_sha256": item.get("raw_request_sha256"),
            "rendered_request_sha256": item.get("rendered_request_sha256"),
            "parent_invocation_id": item.get("parent_invocation_id"),
            **({"target_binding_adapter": receipt.get("target_binding_adapter")}
               if receipt.get("target_binding_adapter") is not None else {}),
            "body_sha256": body_sha256, "body": body,
            "source_kind": "provider-expanded-context-only-same-run-read"})
    return attestations


def _coalesce_text_writer_event_aliases(candidates, *, block_id, block_type, body,
                                       body_sha256, writer_call_id, writer_write_node,
                                       writer_name, source_row_sha256, split, source_groups):
    """Coalesce duplicate source rows for one exact writer event in text context only.

    Native records and separately admitted derived-body records can both describe
    the same provider write. They may ground one text input only when the event,
    typed block, body and source scope are identical. This does not merge their
    training targets or recurrence/runtime representations.
    """
    if not candidates:
        return None
    expected_groups = tuple(sorted(set(source_groups or [])))
    aliases = []
    signatures = set()
    for writer in candidates:
        occurrence_nodes = {item.get("writer_write_node") for item in writer.get("write_occurrences", [])
                            if isinstance(item, dict)}
        event_nodes = {writer.get("writer_write_node"), *occurrence_nodes}
        # Every view must bind to the exact call and graph node named by the
        # already authenticated provider read receipt.
        if (writer.get("writer_invocation_id") != writer_call_id
                or writer_write_node not in event_nodes
                or writer.get("writer_split") != split
                or writer.get("writer_source_row_sha256") != source_row_sha256
                or tuple(sorted(set(writer.get("writer_source_groups") or []))) != expected_groups
                or writer.get("write_name") != writer_name
                or writer.get("body_sha256") != body_sha256
                or writer.get("body") != body
                or block_type != "Neuralese<string>"
                or not isinstance(block_id, str) or not block_id.startswith("nz1_")):
            return None
        signature = (block_id, block_type, body_sha256, writer_call_id, writer_write_node,
                     writer.get("write_name"), tuple(writer.get("result_path") or []),
                     writer.get("source_kind"), source_row_sha256, split, expected_groups)
        signatures.add(signature)
        aliases.append({"record_id": writer.get("writer_record_id"),
                        "record_sha256": writer.get("writer_record_sha256"),
                        "source_kind": writer.get("source_kind"),
                        "source_row_sha256": writer.get("writer_source_row_sha256"),
                        "split": writer.get("writer_split"),
                        "source_groups": sorted(set(writer.get("writer_source_groups") or [])),
                        "write_name": writer.get("write_name"),
                        "write_node": writer.get("writer_write_node")})
    if len(signatures) != 1:
        return None
    # Keep a deterministic representative for existing consumers while making
    # every equivalent source-record attribution explicit in the text receipt.
    representative = sorted(candidates, key=lambda item: (str(item.get("writer_record_id")),
                                                           str(item.get("writer_record_sha256"))))[0]
    return {**representative,
            "equivalent_writer_source_records": sorted(aliases,
                key=lambda item: (str(item.get("record_id")), str(item.get("record_sha256"))))}


def _hydrate_tool_argument_blocks(messages, bodies, read_bodies=None):
    """Turn attested block parts nested in tool arguments into ordinary text parts."""
    hydrated = json.loads(json.dumps(messages, ensure_ascii=False))
    used = []
    def replace(value):
        if isinstance(value, dict):
            if value.get("type") == "neuralese" and isinstance(value.get("id"), str):
                block_id = value["id"]
                if block_id not in bodies:
                    raise ValueError(f"tool argument Neuralese block has no hash-bound source: {block_id}")
                used.append(block_id)
                return {"type": "text", "text": bodies[block_id]}
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return value

    for message in hydrated:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, list):
                # Function arguments are pieces of one JSON text string. A
                # Neuralese part inside that string must be quoted and escaped
                # as a JSON string fragment after its exact captured body is
                # rendered; inserting the body verbatim can turn embedded
                # quotes into malformed JSON. The outer renderer rejoins these
                # text parts without changing their order.
                rendered = []
                for part in args:
                    if (isinstance(part, dict) and part.get("type") == "neuralese"
                            and isinstance(part.get("id"), str)):
                        block_id = part["id"]
                        if block_id not in bodies:
                            raise ValueError(f"tool argument Neuralese block has no hash-bound source: {block_id}")
                        used.append(block_id)
                        body = bodies[block_id]
                        escaped = json.dumps(body, ensure_ascii=False)[1:-1]
                        rendered.append({"type": "text", "text": escaped})
                    elif isinstance(part, dict) and part.get("type") == "read":
                        name, body = part.get("name"), part.get("source")
                        if (not isinstance(name, str) or not isinstance(body, str)
                                or body not in (read_bodies or {}).get(name, set())):
                            raise ValueError("tool argument read lacks exact writer/provider body attestation")
                        escaped = json.dumps(body, ensure_ascii=False)[1:-1]
                        rendered.append({"type": "text", "text": escaped})
                    else:
                        rendered.append(replace(part))
                fn["arguments"] = rendered
            elif isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    continue
                changed = replace(parsed)
                if changed != parsed:
                    fn["arguments"] = json.dumps(changed, ensure_ascii=False, separators=(",", ":"))
    return hydrated, sorted(set(used))


def authenticated_crisp_context_messages(record, piece_map, writer_sources):
    """Render one native context using the same authenticated hydration as ordinary gold text."""
    from ..train.trajectories import (authenticated_capture_context_augmentation, crisp_messages,
                                      handover_notes, write_sites)

    split = record.get("split")
    groups = sorted(set(g for g in (record.get("source_groups") or []) if isinstance(g, str) and g))
    if split not in ("train", "test") or not groups:
        raise ValueError("context record lacks an eligible split or factual source groups")
    notes = handover_notes(record)
    provider_attestations = _attested_provider_expanded_reads(
        record, writer_sources, split=split, source_groups=groups)
    provider_bodies = {item["block_id"]: item for item in provider_attestations
                       if isinstance(item.get("body"), str)}
    neuralese_bodies, context_attestations = _attested_neuralese_message_bodies(
        record, writer_sources, split=split, source_groups=groups,
        authenticated_context_bodies=provider_bodies)
    provider_ids = {item.get("block_id") for item in provider_attestations}
    context_attestations = [item for item in context_attestations
                            if not (item.get("source_kind") == "provider_expanded_context_only_input"
                                    and item.get("block_id") in provider_ids)]
    for item in provider_attestations:
        item.pop("body", None)
    context_attestations.extend(provider_attestations)
    read_bodies = {}
    for name, candidates in writer_sources.items():
        for candidate in candidates:
            if candidate.get("writer_split") == split and isinstance(candidate.get("body"), str):
                read_bodies.setdefault(candidate.get("write_name"), set()).add(candidate["body"])
    for item in provider_attestations:
        if isinstance(item.get("write_name"), str) and isinstance(item.get("body"), str):
            read_bodies.setdefault(item["write_name"], set()).add(item["body"])
    message_inputs, _ = _hydrate_tool_argument_blocks(record.get("messages") or [], neuralese_bodies,
                                                       read_bodies=read_bodies)
    messages = crisp_messages(message_inputs, piece_map, notes, neuralese_bodies=neuralese_bodies)

    by_digest = {}
    for _, _, _, writer_name in write_sites(record):
        augmentation, digest = authenticated_capture_context_augmentation(record, writer_name)
        if augmentation is None:
            continue
        if not isinstance(digest, str):
            raise ValueError("capture context augmentation lacks its digest")
        content = augmentation.get("content")
        if not isinstance(content, str):
            raise ValueError("capture context augmentation digest mismatch")
        prefix, separator, trailer = content.rpartition("\ncontext_augmentation_sha256=")
        if not separator or trailer != digest or _sha(prefix.encode("utf-8")) != digest:
            raise ValueError("capture context augmentation digest mismatch")
        item = by_digest.setdefault(digest, {
            "message": augmentation, "writer_names": [],
            "message_sha256": _sha(_canonical(augmentation).encode("utf-8")),
        })
        if writer_name not in item["writer_names"]:
            item["writer_names"].append(writer_name)
    messages.extend(item["message"] for item in by_digest.values())
    augmentation_attestations = [
        {"writer_names": item["writer_names"], "augmentation_sha256": digest,
         "message_sha256": item["message_sha256"]}
        for digest, item in by_digest.items()]
    return messages, context_attestations, augmentation_attestations


def native_gold_document(tokenizer, messages, target, tools):
    return _native_gold_render(tokenizer,[*messages,target],tools)


def native_gold_packet(tokenizer, messages, target, tools):
    """Render the exact serving prompt followed by the target assistant reply.

    The target boundary is the prompt's token length, not a longest-common-
    prefix between two independently rendered conversations. Appending a new
    target can change how earlier assistant turns render, so LCP is not a
    reliable task-target mask.
    """
    from ..serve.chat import assistant_reply_segments, render_messages, split_escaped
    if list(_message_neuralese_ids(target)):
        raise ValueError("ordinary gold text target contains unresolved neuralese blocks")

    def apply_template(turns, schemas, *, generation):
        return tokenizer.apply_chat_template(
            turns, tools=schemas or None, tokenize=False,
            add_generation_prompt=generation)

    prompt = render_messages(
        messages, tools,
        lambda turns, schemas: apply_template(turns, schemas, generation=True),
        specials=(*tokenizer.all_special_tokens, "<|neuralese|>", "<|/neuralese|>"))
    if prompt.blocks:
        raise ValueError("ordinary gold text prompt contains unresolved neuralese blocks")
    reply_runs = assistant_reply_segments(
        lambda turns, generation: apply_template(turns, tools, generation=generation),
        target, specials=(*tokenizer.all_special_tokens, "<|neuralese|>", "<|/neuralese|>"))
    if reply_runs is None:
        raise ValueError("assistant target does not continue the serving generation prompt")

    text, ids = [], []
    for segment in prompt.segments:
        for run, escaped in split_escaped(segment, prompt.escape_nonce):
            text.append(run)
            ids.extend(tokenizer(run, add_special_tokens=False,
                                 split_special_tokens=escaped)["input_ids"])
    boundary = len(ids)
    for run, escaped in reply_runs:
        text.append(run)
        ids.extend(tokenizer(run, add_special_tokens=False,
                             split_special_tokens=escaped)["input_ids"])
    if boundary >= len(ids):
        raise ValueError("gold assistant target adds no native token suffix")
    return "".join(text), ids, boundary


def gold_text_rows(records: Iterable[Mapping[str, Any]], pieces: Mapping[str, str] | Iterable[Mapping[str, Any]], *, tokenizer,
                   require_independent_splits: bool = True):
    """Return ``(rows, receipt, omissions, provenance)`` for renderer-qualified records.

    Each row has the shared ``text_warmup.load_text_rows`` schema. Text is a
    native chat rendering of crisp messages plus the actual gold target,
    preserving roles and tool schemas/calls. Pretokenized IDs preserve the
    serving distinction between template structure and quoted content markers. Tool
    calls are never executed. Pieces resolve soft references; only explicit
    handover/read source values resolve handovers.
    """
    return _gold_text_rows(records, pieces, tokenizer=tokenizer, preview_only=False,
                           require_independent_splits=require_independent_splits)


def gold_text_preview_rows(records: Iterable[Mapping[str, Any]], pieces: Mapping[str, str] | Iterable[Mapping[str, Any]], *, tokenizer):
    """Render explicitly held review records through the same validated renderer.

    This API creates review artifacts only. It does not set admission fields,
    and its receipt is explicitly ineligible for training.
    """
    return _gold_text_rows(records, pieces, tokenizer=tokenizer, preview_only=True,
                           require_independent_splits=True)


def _gold_text_rows(records, pieces, *, tokenizer, preview_only, require_independent_splits=True):
    from ..train.trajectories import crisp_messages, handover_notes

    fingerprint = tokenizer_fingerprint(tokenizer)
    record_rows = list(records)
    piece_map = (dict(pieces) if isinstance(pieces, Mapping) else
                 {p["name"]: p["text"] for p in pieces
                  if isinstance(p.get("name"), str) and isinstance(p.get("text"), str)})
    prepared, omitted = [], []
    source_hashes, writer_sources = authenticated_context_writer_sources(
        record_rows, preview_only=preview_only)
    for record in record_rows:
        rid = record.get("id", "")
        admitted = (not preview_only
                    and record.get("training_admission", {}).get("approved") is True)
        derived_write = (_authenticated_derived_semantic_text_write(record)
                         if record.get("derived_target") is not None else None)
        if record.get("derived_target") is not None and derived_write is None:
            omitted.append({"id": rid, "reason": "invalid_derived_semantic_text_write_receipt"})
            continue
        held_preview = (preview_only is True
                        and (record.get("review_disposition") == "held_for_root_review"
                             or (derived_write is not None and record.get("review_disposition") ==
                                 "held_derived_equivalent_typed_text_target"))
                        and record.get("training_admission", {}).get("approved") is not True)
        if not (admitted or held_preview):
            omitted.append({"id": rid, "reason": "not_approved_sft_record" if not preview_only
                            else "not_explicitly_held_review_record"})
            continue
        split = record.get("split")
        if split not in ("train", "test"):
            omitted.append({"id": rid, "reason": "invalid_or_missing_split"})
            continue
        groups = sorted(set(g for g in (record.get("source_groups") or []) if isinstance(g, str) and g))
        if not groups:
            omitted.append({"id": rid, "reason": "missing_factual_source_groups"})
            continue
        try:
            messages, context_attestations, capture_augmentation_attestations = (
                authenticated_crisp_context_messages(record, piece_map, writer_sources))
            notes = handover_notes(record)
            target = crisp_messages([record["target"]], piece_map, notes)[0]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            omitted.append({"id": rid, "reason": "unresolved_or_malformed_crisp_reference", "detail": str(exc)[:240]})
            continue
        try:
            text, token_ids, suffix_start = native_gold_packet(tokenizer, messages, target, record.get("tools") or [])
        except (KeyError, TypeError, ValueError) as exc:
            omitted.append({"id": rid, "reason": "native_chat_render_failed", "detail": str(exc)[:240]})
            continue
        if not text.strip():
            omitted.append({"id": rid, "reason": "empty_native_turn"})
            continue
        prepared.append({"text": text, "token_ids": token_ids, "tokenizer_sha256": fingerprint,
                         "supervised_suffix_start": suffix_start,
                         "split": split, "source_groups": groups, "id": rid,
                         "neuralese_context_attestations": [dict(item, reader_record_id=rid)
                                                            for item in context_attestations],
                         "capture_context_augmentation_attestations": capture_augmentation_attestations})

    train_groups = {g for row in prepared if row["split"] == "train" for g in row["source_groups"]}
    test_groups = {g for row in prepared if row["split"] == "test" for g in row["source_groups"]}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(f"factual source groups cross train/test: {overlap[:20]}")
    test_texts = {row["text"] for row in prepared if row["split"] == "test"}
    n_before = len(prepared)
    prepared = [row for row in prepared if not (row["split"] == "train" and row["text"] in test_texts)]
    excluded_train_held = n_before - len(prepared)
    dedup = {}
    for row in prepared:
        key = row["split"], row["text"], tuple(row["token_ids"]),row['supervised_suffix_start']
        if key not in dedup:
            dedup[key] = {**row, "source_record_ids": [row["id"]]}
        else:
            representative = dedup[key]
            representative["source_groups"] = sorted(set(representative["source_groups"] + row["source_groups"]))
            representative["source_record_ids"].append(row["id"])
            representative["neuralese_context_attestations"].extend(row["neuralese_context_attestations"])
            representative["capture_context_augmentation_attestations"].extend(
                row["capture_context_augmentation_attestations"])
    rows = list(dedup.values())
    same_split_dupes = n_before - excluded_train_held - len(rows)
    if (not preview_only and require_independent_splits and
            (not any(row["split"] == "train" for row in rows) or
             not any(row["split"] == "test" for row in rows))):
        raise ValueError("nonempty independent train and held text required")

    provenance = [{"id": row["id"], "split": row["split"], "source_groups": row["source_groups"],
                   "source_record_sha256": source_hashes[row["id"]],
                   "source_records": [{"id": rid, "sha256": source_hashes[rid]} for rid in row["source_record_ids"]],
                   "text_sha256": _sha(row["text"].encode("utf-8")),
                   "token_ids_sha256": _sha(_canonical(row["token_ids"]).encode("utf-8")),
                   "tokenizer_sha256": fingerprint,
                   "neuralese_context_attestations": row["neuralese_context_attestations"],
                   "capture_context_augmentation_attestations": row["capture_context_augmentation_attestations"]}
                  for row in rows]
    for row,entry in zip(rows,provenance):
        entry['supervised_suffix_start']=row['supervised_suffix_start']
    omissions_bytes = "".join(_canonical(row) + "\n" for row in omitted).encode("utf-8")
    provenance_bytes = "".join(_canonical(row) + "\n" for row in provenance).encode("utf-8")
    receipt = {
        "format": ("natlang.gold_text_preview_receipt/1" if preview_only
                   else "natlang.gold_text_packet_receipt/1"),
        "status": "held-review-only" if preview_only else "rendered",
        "review_only": bool(preview_only),
        "sft_eligible": not preview_only,
        "policy": ("explicitly held review records only; output is a non-trainable preview with no admission effect; "
                   if preview_only else "approved SFT records only; ") +
                  "deterministic crisp rendering from supplied pieces and explicit handover notes; exact named Neuralese reader-context blocks hydrate only from a unique approved same-split writer source sharing a source group or source-row hash, or from an explicit exact provider-expanded runtime read receipt bound to body, type, request, graph and source row; context-only receipts never create a writer target or recurrence edge; typed eval-finish marker outputs are rendered from validated exact code sidecars; hash-bound full capture snapshots omitted from historical previews are supplied only as separately labeled same-invocation context augmentations; hydrated context and capture augmentation never create separate target rows; duplicate identical augmentations are emitted once per target; complete source-group split retained; train copies of held complete documents excluded; serving generation prompt and assistant reply rendered separately with native chat content escaping; no tools executed",
        "rendering": "natlang.native_gold_chat/3", "tokenizer_sha256": fingerprint,
        "supervision": "serving generation prompt is context-only; loss begins at the exact token length of that prompt and covers the assistant_reply-rendered target; no historical tool outputs or prior assistant turns are target tokens",
        "ordinary_text_stage_only": True, "task_or_trajectory_admission_granted": False,
        "training_admission_granted": False,
        "documents": len(rows), "train_documents": sum(row["split"] == "train" for row in rows),
        "test_documents": sum(row["split"] == "test" for row in rows),
        "omitted_records": len(omitted), "excluded_train_exact_held_complete_documents": excluded_train_held,
        "duplicate_same_split_documents_deduplicated": same_split_dupes,
        "hash_bound_reader_context_blocks": sum(len(row["neuralese_context_attestations"]) for row in rows),
        "unresolved_omissions": omitted,
        "omissions_jsonl_sha256": _sha(omissions_bytes),
        "provenance_jsonl_sha256": _sha(provenance_bytes),
    }
    return rows, receipt, omitted, provenance
