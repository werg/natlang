#!/usr/bin/env python3
"""Build review-only derived-body/reader edges and optional held R candidates.

Each emitted edge pairs an independently admitted derived body target with an
actual reader action, preserving the source graph and labeling any proposed
learned writer-to-reader relation as counterfactual and nonhistorical. Optional
R output remains unadmitted pending separate review.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.jsonio import canonical_json_bytes as canonical  # noqa: E402


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_json(value):
    return sha_bytes(canonical(value))


def sha_js_json(value):
    """Match the upstream Node converter's SHA-256(JSON.stringify(value))."""
    return sha_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def artifact_path(path):
    value = Path(path).as_posix()
    prefix = "/home/werg/natlang/"
    return value[len(prefix):] if value.startswith(prefix) else value


def classify_reader_action(row, producer_body):
    calls = (row.get("target") or {}).get("tool_calls") or []
    if len(calls) != 1:
        return {"kind": "multiple-or-no-tool-calls", "state_update": "unreviewed"}
    function = (calls[0].get("function") or {})
    name = function.get("name")
    if name == "return_result":
        try:
            args = json.loads(function.get("arguments", "{}"))
        except (TypeError, json.JSONDecodeError):
            return {"kind": "unparsed-return-result", "state_update": "unreviewed"}
        value = args.get("value")
        if isinstance(value, str) and value == producer_body:
            return {
                "kind": "direct-exact-producer-body-return-result",
                "value_sha256": sha_bytes(value.encode("utf-8")),
                "value_equals_exact_block_body": True,
                "state_update": False,
            }
        if "value" not in args:
            return {"kind": "status-only-return-result", "state_update": False}
        if isinstance(value, str):
            return {"kind": "string-return-result-other-than-exact-block-body",
                    "value_sha256": sha_bytes(value.encode("utf-8")),
                    "value_equals_exact_block_body": False, "state_update": "unreviewed"}
        return {"kind": "structured-return-result", "state_update": "unreviewed"}
    if name in {"read_code", "read_page"}:
        return {"kind": "context-read-only", "state_update": False}
    return {"kind": name or "unknown-tool-action", "state_update": "unreviewed"}


def _replace_message_block_with_read(value, block_id, read_name, body):
    """Replace only exact typed body references with an explicit proposed read port."""
    if isinstance(value, dict):
        if value.get("type") == "neuralese" and value.get("id") == block_id:
            return {"type": "read", "name": read_name, "source": body}, 1
        changed, occurrences = {}, 0
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    parsed = json.loads(child)
                except json.JSONDecodeError:
                    changed[key] = child
                else:
                    replacement, count = _replace_message_block_with_read(parsed, block_id, read_name, body)
                    changed[key] = (json.dumps(replacement, ensure_ascii=False, separators=(",", ":"))
                                    if count else child)
                    occurrences += count
            else:
                replacement, count = _replace_message_block_with_read(child, block_id, read_name, body)
                changed[key] = replacement
                occurrences += count
        return changed, occurrences
    if isinstance(value, list):
        replaced, occurrences = [], 0
        for child in value:
            replacement, count = _replace_message_block_with_read(child, block_id, read_name, body)
            replaced.append(replacement); occurrences += count
        return replaced, occurrences
    return value, 0


def materialize_counterfactual_recurrence_rows(edges, derived_rows, reader_rows):
    """Build held R rows pairing approved derived body targets with admitted readers.

    The writer's body is a separately admitted text target projected as a Neuralese
    write. Each reader keeps its exact sampled target and changes only the matching
    authenticated typed input block into a read from that proposed writer. No row
    created here inherits training admission.
    """
    selected = [edge for edge in edges
                if edge.get("observed_reader", {}).get("root_action_disposition", {}).get("status")
                == "root-admitted-action"]
    writers, readers = {}, {}
    relation_ids = []
    for edge in selected:
        producer = edge["producer"]
        observed = edge["observed_reader"]
        producer_id = producer["derived_native_record_id"]
        reader_id = observed["native_record_id"]
        row_entry = derived_rows.get(producer_id)
        if not row_entry:
            raise ValueError(f"counterfactual producer row is missing: {producer_id}")
        derived = row_entry["row"] if isinstance(row_entry, dict) and "row" in row_entry else row_entry
        target = copy.deepcopy(derived.get("target"))
        calls = target.get("tool_calls") if isinstance(target, dict) else None
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError(f"derived target is not a single return_result action: {producer_id}")
        function = calls[0].get("function") or {}
        if function.get("name") != "return_result":
            raise ValueError(f"derived target is not return_result: {producer_id}")
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"derived return_result arguments are malformed: {producer_id}") from exc
        body = arguments.get("value")
        block_id = producer.get("writer_block_id")
        body_hash = sha_bytes(body.encode("utf-8")) if isinstance(body, str) else None
        if (not isinstance(block_id, str) or not block_id.startswith("nz1_")
                or body_hash != producer.get("body_sha256")
                or derived.get("derived_target", {}).get("exact_body_sha256") != body_hash
                or producer.get("target_role") != "approved-derived-body-supervision-not-sampled-action"):
            raise ValueError(f"derived body target differs from its exact approved producer: {producer_id}")
        write_name = "soft-state:" + block_id
        writer_id = "counterfactual-writer:" + producer_id
        prior_writer = writers.get(producer_id)
        if prior_writer is None:
            writer = copy.deepcopy(derived)
            writer["id"] = writer_id
            writer_target = copy.deepcopy(target)
            writer_arguments = dict(arguments)
            writer_arguments["value"] = {"$write": {"name": write_name,
                "type": "Neuralese<string>", "source": body}}
            writer_target["tool_calls"][0]["function"]["arguments"] = json.dumps(
                writer_arguments, ensure_ascii=False, separators=(",", ":"))
            writer["target"] = writer_target
            writer["training_admission"] = {"approved": False,
                "kind": "counterfactual-writer-view-pending-root-review"}
            writer["counterfactual_recurrence_view"] = {
                "schema": "natlang.counterfactual-recurrence-view/1",
                "role": "derived-body-as-proposed-learned-writer",
                "relation_ids": [], "write_name": write_name, "block_id": block_id,
                "body_sha256": body_hash,
                "body_admission_receipt": edge["producer"]["root_body_admission_receipt"],
                "original_eval_hidden_states_equivalent": False,
                "original_learned_writer_state_observed": False,
                "recurrence_admission": False, "runtime_qualification": False,
                "training_admission": False}
            writers[producer_id] = writer
        elif prior_writer["counterfactual_recurrence_view"]["body_sha256"] != body_hash:
            raise ValueError(f"producer has inconsistent bodies across reader edges: {producer_id}")

        reader_entry = reader_rows.get(reader_id)
        if not reader_entry:
            raise ValueError(f"observed reader row is missing: {reader_id}")
        reader = reader_entry["row"] if isinstance(reader_entry, dict) and "row" in reader_entry else reader_entry
        # Observed edge target pins use the sorted-key native-row digest
        # (sha_json); sha_js_json is reserved for the materializer's derived
        # target receipt, which deliberately pins insertion-order JSON.stringify.
        if sha_json(reader.get("target") or {}) != observed.get("target_sha256"):
            raise ValueError(f"reader target differs from exact admitted action: {reader_id}")
        receipt_matches = [receipt for receipt in (reader.get("source_ref", {}).get(
            "provider_expanded_read_contexts") or []) if (receipt.get("block") or {}).get("id") == block_id]
        if len(receipt_matches) != 1:
            raise ValueError(f"reader lacks one exact source read receipt for block: {reader_id}")
        receipt = receipt_matches[0]
        actual_body = (receipt.get("block") or {}).get("body")
        if (actual_body != body or sha_bytes(actual_body.encode("utf-8")) != body_hash
                or (receipt.get("block_read") or {}).get("node") != observed.get("reader_node")
                or (receipt.get("model_turn") or {}).get("node") != observed.get("model_turn_node")):
            raise ValueError(f"reader source body/ports differ from exact graph edge: {reader_id}")
        reader_id_new = "counterfactual-reader:" + reader_id
        if reader_id_new not in readers:
            projected = copy.deepcopy(reader)
            original_messages_sha = sha_js_json(reader.get("messages") or [])
            new_messages, occurrences = _replace_message_block_with_read(
                reader.get("messages") or [], block_id, write_name, body)
            expected_occurrences = receipt.get("context_occurrences")
            if type(expected_occurrences) is not int or occurrences != expected_occurrences or occurrences < 1:
                raise ValueError(f"reader input port replacement count mismatch: {reader_id}")
            projected["id"] = reader_id_new
            projected["messages"] = new_messages
            projected["training_admission"] = {"approved": False,
                "kind": "counterfactual-reader-view-pending-root-review"}
            projected["counterfactual_recurrence_view"] = {
                "schema": "natlang.counterfactual-recurrence-view/1",
                "role": "observed-reader-with-proposed-learned-writer-input",
                "relation_ids": [], "read_name": write_name, "block_id": block_id,
                "body_sha256": body_hash,
                "original_native_record_id": reader_id,
                "original_messages_sha256": original_messages_sha,
                "counterfactual_messages_sha256": sha_js_json(new_messages),
                "reader_action_admission_receipt": edge["observed_reader"]["root_action_disposition"],
                "original_target_sha256": observed["target_sha256"],
                "original_source_graph_edge": observed["observed_source_graph_edge"],
                "read_inputs": observed["read_inputs"], "model_turn_inputs": observed["model_turn_inputs"],
                "original_eval_hidden_states_equivalent": False,
                "original_learned_writer_state_observed": False,
                "recurrence_admission": False, "runtime_qualification": False,
                "training_admission": False}
            readers[reader_id] = projected
        elif readers[reader_id]["counterfactual_recurrence_view"]["read_name"] != write_name:
            raise ValueError(f"reader has conflicting proposed producer ports: {reader_id}")
        relation = edge["proposal_id"]
        writers[producer_id]["counterfactual_recurrence_view"]["relation_ids"].append(relation)
        readers[reader_id]["counterfactual_recurrence_view"]["relation_ids"].append(relation)
        relation_ids.append(relation)
    for row in [*writers.values(), *readers.values()]:
        row["counterfactual_recurrence_view"]["relation_ids"].sort()
    return [*writers.values(), *readers.values()]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    found = {}
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            identity = row.get("id")
            if not isinstance(identity, str) or identity in found:
                raise ValueError(f"{path}:{line_number}: missing or duplicate row id")
            found[identity] = {"row": row, "line_sha256": sha_bytes(raw),
                               "line_number": line_number}
    return found


def events_by_node(trace_path):
    events = {}
    for line_number, raw in enumerate(trace_path.read_bytes().splitlines(), 1):
        event = json.loads(raw)
        node = event.get("node")
        if node:
            if node in events:
                raise ValueError(f"{trace_path}:{line_number}: duplicate graph node {node}")
            events[node] = event
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True,
                        help="exact candidate JSON; repeat once per admitted derived body")
    parser.add_argument("--admission-index", type=Path, required=True)
    parser.add_argument("--derived-native", type=Path, required=True,
                        help="explicit compact native delta containing the admitted derived rows")
    parser.add_argument("--reader-native", type=Path, action="append", required=True,
                        help="full native source export; repeat for each source program")
    parser.add_argument("--root-action-admission", type=Path, action="append", default=[],
                        help="current per-action root admission receipt; repeat as needed")
    parser.add_argument("--writer-stage-review", type=Path, action="append", default=[],
                        help="current writer-stage disposition review; repeat as needed")
    parser.add_argument("--out", type=Path, required=True,
                        help="review-only JSONL edges")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="review/provenance manifest")
    parser.add_argument("--recurrence-out", type=Path,
                        help="optional held counterfactual R rows: derived writer targets plus selected reader views")
    parser.add_argument("--recurrence-pieces-out", type=Path,
                        help="required with --recurrence-out; empty when all proposed body sources stay inline")
    args = parser.parse_args()
    if (args.recurrence_out is None) != (args.recurrence_pieces_out is None):
        raise ValueError("--recurrence-out and --recurrence-pieces-out must be supplied together")
    if args.out.resolve() == args.manifest.resolve():
        raise ValueError("candidate output and manifest must be distinct paths")
    output_paths = [args.out, args.manifest]
    if args.recurrence_out is not None:
        output_paths.extend([args.recurrence_out, args.recurrence_pieces_out])
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("proposal output paths must be distinct")
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite an existing proposal output or manifest")

    index = load_json(args.admission_index)
    if index.get("schema") != "natlang.root-derived-body-admission-index/1":
        raise ValueError("unsupported root body-admission index")
    admission_by_proposal = {}
    for item in index.get("admissions", []):
        receipt_path = Path(item["path"])
        receipt_bytes = receipt_path.read_bytes()
        if sha_bytes(receipt_bytes) != item.get("sha256"):
            raise ValueError(f"body admission pin mismatch: {receipt_path}")
        receipt = json.loads(receipt_bytes)
        if (receipt.get("training_admission") is not True
                or receipt.get("decision") != "admit-derived-observed-text-writer-body"):
            raise ValueError(f"body receipt does not admit a derived body: {receipt_path}")
        proposal_id = receipt.get("proposal_id")
        if not proposal_id or proposal_id in admission_by_proposal:
            raise ValueError("missing or duplicate proposal in body admission index")
        admission_by_proposal[proposal_id] = {
            "path": receipt_path.as_posix(), "sha256": item["sha256"],
            "receipt": receipt,
        }
    if len(args.candidate) != index.get("count"):
        raise ValueError("explicit candidate list length differs from admission index")

    reader_rows = {}
    reader_source_by_id = {}
    reader_input_pins = {}
    for path in args.reader_native:
        raw = path.read_bytes()
        reader_input_pins[path.as_posix()] = {"sha256": sha_bytes(raw), "bytes": len(raw)}
        parsed_rows = read_rows(path)
        overlap = set(reader_rows) & set(parsed_rows)
        if overlap:
            raise ValueError(f"duplicate reader native IDs across inputs: {sorted(overlap)[:3]}")
        reader_rows.update(parsed_rows)
        reader_source_by_id.update({record_id: path.as_posix() for record_id in parsed_rows})

    root_action_admissions = {}
    action_disposition_pins = {}
    for path in args.root_action_admission:
        raw = path.read_bytes()
        action_disposition_pins[path.as_posix()] = {"sha256": sha_bytes(raw), "bytes": len(raw)}
        receipt = json.loads(raw)
        if receipt.get("schema") != "natlang.root-per-action-training-admission/1":
            raise ValueError(f"unsupported root action admission receipt: {path}")
        for row in receipt.get("rows", []):
            native_id = row.get("native_id")
            if not native_id or native_id in root_action_admissions:
                raise ValueError(f"missing or duplicate root action disposition: {native_id}")
            root_action_admissions[native_id] = {
                "decision": row.get("decision"),
                "training_admission": row.get("training_admission") is True,
                "receipt_path": path.as_posix(),
                "receipt_sha256": sha_bytes(raw),
            }
    writer_stage_dispositions = {}
    for path in args.writer_stage_review:
        raw = path.read_bytes()
        action_disposition_pins[path.as_posix()] = {"sha256": sha_bytes(raw), "bytes": len(raw)}
        review = json.loads(raw)
        for row in review.get("writer_stages", []):
            native_id = row.get("native_record_id")
            if not native_id:
                continue
            value = {"status": row.get("status"), "rationale": row.get("rationale"),
                     "review_path": path.as_posix(), "review_sha256": sha_bytes(raw)}
            previous = writer_stage_dispositions.get(native_id)
            if previous and previous != value:
                raise ValueError(f"conflicting writer-stage dispositions: {native_id}")
            writer_stage_dispositions[native_id] = value

    delta_path = args.derived_native
    delta_rows = read_rows(delta_path)
    edges = []
    producer_ids = set()
    result_pins = {}
    for candidate_path in args.candidate:
        candidate_bytes = candidate_path.read_bytes()
        candidate = json.loads(candidate_bytes)
        proposal_id = candidate.get("proposal_id")
        admission = admission_by_proposal.get(proposal_id)
        if not admission:
            raise ValueError(f"candidate lacks root body admission: {proposal_id}")
        receipt = admission["receipt"]
        candidate_pin = receipt.get("input_pins", {}).get(artifact_path(candidate_path))
        if (not candidate_pin or candidate_pin.get("sha256") != sha_bytes(candidate_bytes)
                or candidate_pin.get("bytes") != len(candidate_bytes)):
            raise ValueError(f"candidate bytes are not pinned by root body admission: {proposal_id}")
        source = candidate.get("source_result") or {}
        result_path = Path(source.get("path", ""))
        result_bytes = result_path.read_bytes()
        if sha_bytes(result_bytes) != source.get("sha256"):
            raise ValueError(f"source result pin mismatch: {result_path}")
        trace_path = result_path.with_name(result_path.name.replace(".result.json", ".trace.jsonl"))
        trace_bytes = trace_path.read_bytes()
        # `source.trace_sha256` is the native materializer's in-memory trace
        # digest. It is intentionally distinct from the JSONL export digest.
        raw_trace_sha = sha_bytes(trace_bytes)
        result_pins[result_path.as_posix()] = {"sha256": sha_bytes(result_bytes)}
        result_pins[trace_path.as_posix()] = {
            "sha256": raw_trace_sha,
            "materializer_trace_digest": source.get("trace_sha256"),
            "digest_convention": "in-memory materializer trace digest is distinct from exported JSONL file SHA-256",
        }
        trace_nodes = events_by_node(trace_path)
        writer = candidate.get("writer_graph_event") or {}
        writer_node = writer.get("node")
        observed_writer = trace_nodes.get(writer_node) or {}
        if (any(observed_writer.get(key) != value for key, value in writer.items())
                or observed_writer.get("trace_role") != "child"
                or observed_writer.get("trace_invocation_id") != source.get("invocation_id")):
            raise ValueError(f"candidate writer event not found verbatim in pinned trace: {proposal_id}")
        body = candidate.get("exact_body")
        body_sha = candidate.get("exact_body_sha256")
        origin = candidate.get("body_origin") or {}
        if (not isinstance(body, str) or sha_bytes(body.encode("utf-8")) != body_sha
                or body_sha != origin.get("body_sha256")
                or body_sha != receipt.get("exact_body_sha256")
                or writer.get("result_type") != "Neuralese<string>"
                or writer.get("block") != origin.get("block_id")):
            raise ValueError(f"derived body origin/hash/type mismatch: {proposal_id}")
        derived_row = next((value["row"] for value in delta_rows.values()
                            if value["row"].get("derived_target", {}).get("proposal_id") == proposal_id), None)
        if not derived_row:
            raise ValueError(f"admitted derived native target missing: {proposal_id}")
        derived_id = derived_row["id"]
        if derived_id in producer_ids:
            raise ValueError(f"duplicate producer target: {derived_id}")
        producer_ids.add(derived_id)
        if (derived_row.get("derived_target", {}).get("exact_body_sha256") != body_sha
                or derived_row.get("training_admission", {}).get("approved") is not True
                or derived_row.get("split") != receipt.get("split")
                or receipt.get("source_group") not in (derived_row.get("source_groups") or [])):
            raise ValueError(f"derived native row does not match its admitted body/source: {proposal_id}")
        derived_supervision = candidate.get("derived_supervision") or {}
        expected_target = derived_supervision.get("target")
        expected_target_sha = derived_supervision.get("target_sha256")
        actual_target_sha = sha_js_json(expected_target)
        if (expected_target is None or derived_row.get("target") != expected_target
                or expected_target_sha != actual_target_sha
                or derived_row.get("derived_target", {}).get("target_sha256") != actual_target_sha
                or derived_row.get("training_admission", {}).get("exact_target_sha256") != actual_target_sha
                or receipt.get("derived_target_sha256") != actual_target_sha):
            raise ValueError(f"native target differs from exact admitted candidate/receipt target: {proposal_id}")
        original_target = derived_supervision.get("original_target")
        original_messages = derived_supervision.get("original_messages")
        derived_metadata = derived_row.get("derived_target", {})
        if (derived_row.get("messages") != original_messages
                or derived_metadata.get("original_target") != original_target
                or derived_metadata.get("original_native_record_id") != receipt.get("source_native_record_id")
                or derived_supervision.get("original_target_sha256") != sha_js_json(original_target)
                or derived_metadata.get("original_target_sha256") != sha_js_json(original_target)):
            raise ValueError(f"native context/original sampled action differs from pinned candidate: {proposal_id}")
        limits = receipt.get("limits") or {}
        if (limits.get("original_eval_hidden_states_equivalent") is not False
                or limits.get("learned_vectors") is not False
                or limits.get("runtime_gradient_qualification") is not False
                or limits.get("whole_trajectory_admission") is not False):
            raise ValueError(f"root receipt does not preserve derived-target scope limits: {proposal_id}")
        producer_context_sha = sha_json(derived_row.get("messages") or [])
        producer_target_sha = sha_json(derived_row.get("target") or {})

        for binding in candidate.get("downstream_reader_bindings") or []:
            reader_id = binding.get("native_record_id")
            indexed = reader_rows.get(reader_id)
            if not indexed:
                raise ValueError(f"reader native row missing: {reader_id}")
            reader = indexed["row"]
            if (reader.get("source_ref", {}).get("invocation_id") != binding.get("invocation_id")
                    or reader.get("source_ref", {}).get("source_row_sha256") != source.get("source_row_sha256")
                    or reader.get("decision", {}).get("source_raw_response_sha256") != binding.get("source_response_sha256")
                    or sha_json(reader.get("target") or {}) != binding.get("source_action_target_sha256")):
                raise ValueError(f"reader invocation/response/target binding mismatch: {reader_id}")
            if (reader.get("split") != receipt.get("split")
                    or receipt.get("source_group") not in (reader.get("source_groups") or [])):
                raise ValueError(f"reader crosses split/source-group boundary: {reader_id}")
            read_event = trace_nodes.get(binding.get("read_node"))
            turn_event = trace_nodes.get(binding.get("model_turn_node"))
            block_id = origin["block_id"]
            if (not read_event or read_event.get("kind") != "block_read"
                    or read_event.get("call_id") != binding.get("invocation_id")
                    or read_event.get("block") != block_id
                    or read_event.get("inputs") != binding.get("read_inputs")
                    or read_event.get("trace_role") not in {"root", "child"}
                    or read_event.get("trace_invocation_id") != binding.get("invocation_id")
                    or not turn_event or turn_event.get("kind") != "model_turn"
                    or turn_event.get("call_id") != binding.get("invocation_id")
                    or turn_event.get("inputs") != binding.get("model_turn_inputs")
                    or not any(edge.get("node") == binding.get("read_node")
                               and edge.get("block") == block_id for edge in turn_event.get("inputs", []))):
                raise ValueError(f"reader port binding is not authenticated by exact trace graph: {reader_id}")
            source_edge_present = any(
                edge.get("node") == writer_node and edge.get("block") == block_id
                for edge in binding.get("read_inputs", []))
            if not source_edge_present:
                raise ValueError(f"reader does not consume this exact producer node/block: {reader_id}")
            action_review = classify_reader_action(reader, body)

            row = {
                "schema": "natlang.derived-counterfactual-writer-observed-reader-candidate/1",
                "proposal_id": f"counterfactual:{proposal_id}->{reader_id}",
                "relation": "derived-counterfactual-writer-to-observed-reader",
                "source_execution_graph": "authenticated-original-typed-write-to-observed-read/model-turn-edge",
                "learner_pairing": "derived-body-writer-target-paired-with-authentic-observed-reader-context/action",
                "producer": {
                    "derived_native_record_id": derived_id,
                    "root_body_admission_receipt": {"path": admission["path"], "sha256": admission["sha256"]},
                    "source_native_record_id": receipt.get("source_native_record_id"),
                    "source_group": receipt["source_group"], "split": receipt["split"],
                    "writer_block_id": block_id, "body_sha256": body_sha,
                    "derived_target_sha256": derived_row["derived_target"]["target_sha256"],
                    "derived_target_sha256_convention": "sha256 of compact UTF-8 JSON.stringify(target), preserving object insertion order, as used by the frozen JS converter",
                    "source_result_sha256": source["sha256"],
                    "materializer_trace_digest": source["trace_sha256"],
                    "source_trace_jsonl_sha256": raw_trace_sha,
                    "writer_event_node": writer_node,
                    "writer_event_sha256": sha_json(writer),
                    "writer_trace_role": observed_writer.get("trace_role"),
                    "derived_row_messages_sha256": producer_context_sha,
                    "derived_row_target_sha256": producer_target_sha,
                    "target_role": "approved-derived-body-supervision-not-sampled-action",
                },
                "observed_reader": {
                    "native_record_id": reader_id,
                    "native_source_file": reader_source_by_id[reader_id],
                    "native_row_line": indexed["line_number"],
                    "native_row_line_sha256": indexed["line_sha256"],
                    "invocation_id": binding["invocation_id"],
                    "source_row_sha256": reader.get("source_ref", {}).get("source_row_sha256"),
                    "split": reader.get("split"), "source_groups": reader.get("source_groups"),
                    "messages_sha256": sha_json(reader.get("messages") or []),
                    "target_sha256": sha_json(reader.get("target") or {}),
                    "observed_action_form": action_review,
                    "source_response_sha256": binding["source_response_sha256"],
                    "source_action_target_sha256": binding["source_action_target_sha256"],
                    "reader_node": binding["read_node"],
                    "model_turn_node": binding["model_turn_node"],
                    "reader_graph_call_id": read_event.get("call_id"),
                    "reader_trace_role": read_event.get("trace_role"),
                    "observed_source_graph_edge": {
                        "writer_node": writer_node,
                        "reader_node": binding["read_node"],
                        "block_id": block_id,
                        "verified_by_exact_read_inputs": source_edge_present,
                    },
                    "read_inputs": binding["read_inputs"],
                    "model_turn_inputs": binding["model_turn_inputs"],
                    "native_materializer_training_approved": reader.get("training_admission", {}).get("approved") is True,
                    "native_materializer_flag_is_not_reader_semantic_admission": True,
                    "native_action_admission_kind": reader.get("training_admission", {}).get("kind"),
                    "root_action_disposition": (
                        {"status": "held-by-current-writer-stage-review", **writer_stage_dispositions[reader_id]}
                        if writer_stage_dispositions.get(reader_id, {}).get("status", "").startswith("held_")
                        else ({"status": "root-admitted-action", **root_action_admissions[reader_id]}
                              if reader_id in root_action_admissions
                              else {"status": "not-reviewed-by-supplied-root-action-receipts"})
                    ),
                    "failed_action": reader.get("decision", {}).get("failed_action") is True,
                    "soft_edge_admission": (
                        "held-pending-current-action-disposition-review"
                        if writer_stage_dispositions.get(reader_id, {}).get("status", "").startswith("held_")
                        else "pending-individual-reader-semantics-review"
                    ),
                },
                "limits": {
                    "original_eval_hidden_states_equivalent": False,
                    "original_source_typed_writer_execution_observed": True,
                    "original_source_writer_to_reader_graph_edge_observed": True,
                    "original_learned_writer_state_observed": False,
                    "derived_target_is_counterfactual_learned_writer_view": True,
                    "runtime_gradient_qualification": False,
                    "recurrence_admission": False,
                    "reader_action_admission_implied": False,
                },
            }
            edges.append(row)

    if len(producer_ids) != index.get("count") or len(edges) != sum(
            len(json.loads(path.read_text()).get("downstream_reader_bindings") or [])
            for path in args.candidate):
        raise ValueError("producer or reader edge count differs from exact source index")
    recurrence_rows = None
    recurrence_bytes = None
    recurrence_pieces_bytes = None
    if args.recurrence_out is not None:
        recurrence_rows = materialize_counterfactual_recurrence_rows(edges, delta_rows, reader_rows)
        recurrence_bytes = b"".join(canonical(row) + b"\n" for row in recurrence_rows)
        # Current counterfactual views keep authenticated body text inline in the
        # standard $write.source/read.source ports, so they introduce no soft-piece refs.
        recurrence_pieces_bytes = b""
        edges_by_producer = {}
        for edge in edges:
            if edge.get("observed_reader", {}).get("root_action_disposition", {}).get("status") == "root-admitted-action":
                edges_by_producer.setdefault(edge["producer"]["derived_native_record_id"], []).append(edge["proposal_id"])
        row_ids = {row["id"] for row in recurrence_rows}
        expected_writers = {"counterfactual-writer:" + ident for ident in edges_by_producer}
        expected_readers = {"counterfactual-reader:" + row["observed_reader"]["native_record_id"]
                            for row in edges if row.get("observed_reader", {}).get(
                                "root_action_disposition", {}).get("status") == "root-admitted-action"}
        if row_ids != expected_writers | expected_readers:
            raise ValueError("counterfactual R rows do not exactly cover admitted producer/reader views")
    builder_path = Path(__file__).resolve()
    builder_snapshot = args.out.parent / "code-snapshot" / builder_path.name
    if builder_snapshot.exists():
        raise FileExistsError(f"refusing to overwrite builder snapshot: {builder_snapshot}")
    builder_bytes = builder_path.read_bytes()
    out_bytes = b"".join(canonical(row) + b"\n" for row in edges)
    inputs = {
        args.admission_index.as_posix(): {"sha256": sha_bytes(args.admission_index.read_bytes())},
        delta_path.as_posix(): {"sha256": sha_bytes(delta_path.read_bytes())},
        **result_pins,
        **reader_input_pins,
        **action_disposition_pins,
    }
    candidate_pins = {}
    for p in args.candidate:
        candidate_pins[p.as_posix()] = {"sha256": sha_bytes(p.read_bytes())}
    for item in index.get("admissions", []):
        receipt_path = Path(item["path"])
        inputs[receipt_path.as_posix()] = {"sha256": item["sha256"]}
    inputs.update(candidate_pins)
    manifest = {
        "schema": "natlang.derived-counterfactual-writer-reader-proposal-manifest/1",
        "status": "review-only-not-recurrence-admitted",
        "producer_count": len(producer_ids), "observed_reader_binding_count": len(edges),
        "unique_reader_action_count": len({r["observed_reader"]["native_record_id"] for r in edges}),
        "candidate_rows_sha256": sha_bytes(out_bytes), "candidate_rows": args.out.as_posix(),
        "counterfactual_recurrence": ({
            "path": args.recurrence_out.as_posix(), "sha256": sha_bytes(recurrence_bytes),
            "bytes": len(recurrence_bytes), "rows": len(recurrence_rows),
            "writer_rows": sum(row["counterfactual_recurrence_view"]["role"] ==
                                "derived-body-as-proposed-learned-writer" for row in recurrence_rows),
            "reader_rows": sum(row["counterfactual_recurrence_view"]["role"] ==
                                "observed-reader-with-proposed-learned-writer-input" for row in recurrence_rows),
            "training_admission": False, "recurrence_admission": False,
            "original_eval_hidden_states_equivalent": False,
            "pieces": {"path": args.recurrence_pieces_out.as_posix(),
                       "sha256": sha_bytes(recurrence_pieces_bytes), "bytes": len(recurrence_pieces_bytes),
                       "rows": 0, "reason": "all exact source/body strings are inline in standard $write.source and read.source ports"},
            "relations": len([edge for edge in edges if edge.get("observed_reader", {}).get(
                "root_action_disposition", {}).get("status") == "root-admitted-action"]),
        } if recurrence_rows is not None else None),
        "builder": {
            "source_path": builder_path.as_posix(), "source_sha256": sha_bytes(builder_bytes),
            "immutable_snapshot_path": builder_snapshot.as_posix(),
            "immutable_snapshot_sha256": sha_bytes(builder_bytes),
        },
        "inputs": inputs,
        "argv": sys.argv,
        "digest_conventions": {
            "source_jsonl_line_sha256": "sha256 of exact line bytes including terminal LF",
            "derived_row_messages_sha256": "SHA-256 of compact UTF-8 JSON with sorted keys and no insignificant whitespace",
            "derived_row_target_sha256": "SHA-256 of compact UTF-8 JSON with sorted keys and no insignificant whitespace",
            "candidate_rows_sha256": "SHA-256 of canonical compact JSONL bytes emitted by this builder",
        },
        "policy": "Each producer target is separately root-admitted derived body supervision. The source graph proves an actual typed writer block and exact downstream block-read/model-turn ports. The proposed learner view pairs that admitted body target with an actual observed reader context/action, but it does not claim the original eval had learned writer states or matching hidden states. Reader semantics, source action admission, recurrence admission, and runtime qualification remain separate.",
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    # Recheck immediately before mutation, then create every output exclusively.
    if any(path.exists() for path in output_paths) or builder_snapshot.exists():
        raise FileExistsError("refusing to overwrite an existing proposal artifact")
    output_payloads = [(builder_snapshot, builder_bytes), (args.out, out_bytes)]
    if recurrence_rows is not None:
        output_payloads.extend([(args.recurrence_out, recurrence_bytes),
                                (args.recurrence_pieces_out, recurrence_pieces_bytes)])
    output_payloads.append((args.manifest, manifest_bytes))
    for output_path, payload in output_payloads:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    print(json.dumps({"producers": len(producer_ids), "bindings": len(edges),
                      "unique_readers": manifest["unique_reader_action_count"],
                      "rows_sha256": manifest["candidate_rows_sha256"]}))


if __name__ == "__main__":
    main()
