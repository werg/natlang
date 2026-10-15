#!/usr/bin/env python3
"""Convert selected, exact-success teacher traces into ordinary native text graphs.

This static transformation preserves each captured request context and the
teacher's actual next assistant turn. Text-emulated Neuralese remains ordinary
text here; this tool does not invent producer/consumer edges. Such edges need
the separate source-authenticated semantic-pair converter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.neuralese_data.build_native_semantic_pairs_e2e import CompressedRows, provision
from scripts import source_review

from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.data.native_graph_conversion import text_row_graph
from natlang_neuralese.data.text_corpus import native_gold_packet, tokenizer_fingerprint
from natlang_neuralese.model.native_model import base_inventory
from natlang_neuralese.model.shared_state import ROLE_MARKERS


def file_rows(path: Path):
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    yield raw, json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


class GraphWriter(CompressedRows):
    def __init__(self, path):
        super().__init__(path)
        self.tokens, self.max_tokens = 0, 0

    def write(self, graph):
        super().write(graph)
        count = len(graph['nodes'][0]['token_ids'])
        self.tokens += count
        self.max_tokens = max(self.max_tokens, count)

    def finish(self):
        return {**super().finish(), 'tokens': self.tokens, 'max_row_tokens': self.max_tokens}


def current_protection(path):
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'natlang.training-protection-projection/1':
        raise ValueError('current protection needs its exact projection manifest')
    held, quality = set(), set()
    for item in manifest['files']:
        file = (path.parent / item['path']).resolve(strict=True)
        if sha256_file(file) != item['sha256'] or file.stat().st_size != item['bytes']:
            raise ValueError('current protection artifact differs: ' + str(file))
        group_kind = ('held' if item['path'].startswith('held-union/') else
                      'quality' if item['path'] == 'quality-groups.jsonl' else None)
        if group_kind:
            target = held if group_kind == 'held' else quality
            for _, row in file_rows(file):
                target.update(row.get('source_groups', []))
    if len(held) != manifest['held_groups'] or len(quality) != manifest['quality_groups']:
        raise ValueError('current protection group projection is incomplete')
    live_review = source_review.REGISTRY / 'holds.jsonl'
    if sha256_file(live_review) != manifest['source_review_sha256']:
        raise ValueError('protection projection differs from current source reviews')
    pending = source_review.pending_unpinned_source_groups()
    return {'held': held, 'quality': quality, 'pending': pending,
            'proof': {'path': str(path), 'sha256': sha256_file(path),
                'held_groups': len(held), 'quality_groups': len(quality),
                'source_review_sha256': manifest['source_review_sha256']}}


def source_binding(args):
    root = args.source.parent
    transfer_path = root / "transfer-receipt.json"
    selection_path = root / "selection-proof.json"
    global_path = root / "global-protection-join.json"
    evidence_path = root / "evidence-source-v2.json"
    server_path = root / "penny-server-binding.json"
    proof_paths = {"transfer": transfer_path, "selection": selection_path,
                   "global_protection_join": global_path, "evidence_source": evidence_path,
                   "teacher_runtime": server_path}
    proofs = {name: json.loads(path.read_text()) for name, path in proof_paths.items()}
    actual_source_sha = sha256_file(args.source)
    transfer = proofs["transfer"]
    selection = proofs["selection"]
    protection = proofs["global_protection_join"]
    evidence = proofs["evidence_source"]
    teacher = proofs["teacher_runtime"]
    if (transfer.get("schema") != "natlang.verified-teacher-trajectory-transfer/1"
            or transfer.get("combined", {}).get("path") != str(args.source)
            or transfer.get("combined", {}).get("sha256") != actual_source_sha
            or transfer.get("combined", {}).get("bytes") != args.source.stat().st_size
            or transfer.get("training_admission") is not False):
        raise ValueError("transfer receipt does not bind the exact raw teacher trajectory source")
    if (selection.get("schema") != "natlang.native-teacher-task-selection/1"
            or selection.get("status") != "passed"):
        raise ValueError("teacher output lacks its exact passed source-selection proof")
    selected_path = root / "selected-cases.jsonl"
    if sha256_file(selected_path) != selection.get("output_sha256"):
        raise ValueError("selected task packet differs from its immutable selection proof")
    if (protection.get("schema") != "natlang.native-teacher-global-protection-join/1"
            or protection.get("status") != "passed"
            or protection.get("selected_packet_sha256") != selection.get("output_sha256")
            or protection.get("held_group_intersections")
            or protection.get("quality_group_intersections")):
        raise ValueError("teacher selection intersects protected evaluation or quality-held groups")
    if teacher.get("teacher_role") != "independent stronger open teacher; not either student/onpolicy":
        raise ValueError("teacher provenance does not identify an independent teacher")
    selected_ids = set(selection.get("selected_ids", []))
    train_groups = set(selection.get("train_groups", []))
    held_groups = set(selection.get("held_groups", []))
    if train_groups & held_groups:
        raise ValueError("selection proof train/held source groups overlap")
    source_files = {item.get("row_id"): item for item in transfer.get("files", [])
                    if isinstance(item.get("row_id"), str)}
    collections = {}
    for item in transfer.get('files', []):
        if not str(item.get('source', '')).endswith('/teacher-source.json'):
            continue
        local = Path(item['path'])
        if sha256_file(local) != item['sha256'] or local.stat().st_size != item['bytes']:
            raise ValueError('original collection receipt differs from its transfer binding')
        collection = json.loads(local.read_text())
        if (collection.get('schema') != 'natlang.sandbox-strong-teacher-collection/1'
                or collection.get('source_cases_sha256') != selection['output_sha256']
                or canonical_json_sha256_hex(collection.get('protected_selection')) != canonical_json_sha256_hex(selection)
                or canonical_json_sha256_hex(collection.get('teacher')) != canonical_json_sha256_hex(teacher)):
            raise ValueError('collection receipt differs from selected tasks or actual stronger teacher')
        collections[item['source']] = {'path': str(local), 'sha256': item['sha256']}
    if not collections:
        raise ValueError('teacher trajectories need immutable per-batch collection receipts')
    proof_sha = {name: sha256_file(path) for name, path in proof_paths.items()}
    return {"source_sha256": actual_source_sha, "transfer": transfer, "selection": selection,
            "protection": protection, "evidence": evidence, "teacher": teacher,
            "source_files": source_files, "selected_ids": selected_ids,
            "train_groups": train_groups, "held_groups": held_groups,
            "proof_paths": proof_paths, "proof_sha256": proof_sha, "collections": collections,
            "current": current_protection(args.protection_manifest)}


def validate_teacher_row(row, binding):
    program = (row.get("task") or {}).get("program_ir") or {}
    outcome = row.get("outcome") or {}
    if (row.get("version") != "natlang.teacher_trajectory.native/1"
            or program.get("version") != "natlang.program/2"
            or program.get("split") != "train"
            or program.get("id") not in binding["selected_ids"]
            or outcome.get("status") != "done" or outcome.get("accepted") is not True
            or outcome.get("oracle", {}).get("accepted") is not True
            or not outcome.get("checks")
            or any(value is not True for value in outcome["checks"].values())):
        raise ValueError(f"teacher trajectory is not an exact selected train success: {row.get('id')}")
    groups = program.get("source_groups")
    if (not isinstance(groups, list) or not groups or len(groups) != len(set(groups))
            or not set(groups) <= binding["train_groups"] or set(groups) & binding["held_groups"]):
        raise ValueError(f"teacher trajectory lacks exact train-only source-group proof: {row.get('id')}")
    current = binding['current']
    if set(groups) & (current['held'] | current['quality'] | current['pending']):
        raise ValueError('teacher trajectory intersects current held/quality/source-review protection: ' + str(row.get('id')))
    reason = source_review.pending_family_hold_reason(program)
    if reason:
        raise ValueError('teacher trajectory has a pending source family review: ' + str(reason))
    file_receipt = binding["source_files"].get(row.get("id"))
    if not isinstance(file_receipt, dict):
        raise ValueError(f"teacher trajectory lacks its source response-file binding: {row.get('id')}")
    batch_source = str(Path(file_receipt['source']).parents[2] / 'teacher-source.json')
    if batch_source not in binding['collections']:
        raise ValueError('teacher response is not bound to its immutable collection batch')
    trace = row.get("provenance") or {}
    if trace.get("model") != "pennyroyal" or row.get("execution_identity", {}).get("execution_run_id") is None:
        raise ValueError(f"teacher trajectory lacks the expected runtime execution identity: {row.get('id')}")
    if not isinstance(row.get("trajectory"), list) or not row["trajectory"]:
        raise ValueError(f"teacher trajectory has no captured whole-turn contexts: {row.get('id')}")
    return program, file_receipt


def verify_original_row(row, original):
    path = Path(original['path'])
    if not path.is_file() or sha256_file(path) != original.get('sha256'):
        raise ValueError('original accepted response artifact changed: ' + str(path))
    originals = [value for _, value in file_rows(path) if value.get('id') == row['id']]
    if len(originals) != 1 or canonical_json_sha256_hex(originals[0]) != canonical_json_sha256_hex(row):
        raise ValueError('combined teacher trajectory differs from its original response: ' + row['id'])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--protection-manifest", type=Path, required=True)
    parser.add_argument("--model-key", choices=("lfm25", "mellum21"), required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {args.output}")
    binding = source_binding(args)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_tokenizer), trust_remote_code=False,
                                               local_files_only=True)
    source_fingerprint = tokenizer_fingerprint(tokenizer)
    role_ids = provision(tokenizer)
    expanded_fingerprint = tokenizer_fingerprint(tokenizer)
    inventory = base_inventory(args.base_tokenizer)
    model_type = json.loads((args.base_tokenizer / "config.json").read_text())["model_type"]
    model_id = model_type + ":" + canonical_json_sha256_hex(inventory)
    writer = GraphWriter(args.output / "train" / "native-graphs.jsonl.zst")
    held = GraphWriter(args.output / "test" / "native-graphs.jsonl.zst")
    seen_ids = set()
    row_hashes = []
    teacher_turns = 0
    try:
        for raw, row in file_rows(args.source):
            program, original = validate_teacher_row(row, binding)
            source_row_sha = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
            original_path = verify_original_row(row, original)
            row_hashes.append({"id": row["id"], "sha256": source_row_sha,
                               "program_id": program["id"], "source_groups": sorted(set(program["source_groups"])),
                               "original_response_path": str(original_path),
                               "original_response_sha256": original["sha256"]})
            for index, turn in enumerate(row["trajectory"]):
                if turn.get("phase") != "action":
                    raise ValueError(f"unknown captured teacher turn phase on {row['id']}[{index}]")
                context = turn.get("context")
                assistant = turn.get("assistant")
                raw_calls = (turn.get("model_response") or {}).get("raw_calls")
                tools = turn.get("tools_offered")
                if (not isinstance(context, list) or not isinstance(assistant, dict)
                        or not isinstance(raw_calls, list) or not isinstance(tools, list)):
                    raise ValueError(f"teacher turn lacks exact raw request/response structure: {row['id']}[{index}]")
                target = {"role": "assistant", "content": assistant.get("content", ""),
                          "tool_calls": raw_calls}
                if isinstance(assistant.get("reasoning"), str) and assistant["reasoning"]:
                    target["reasoning_content"] = assistant["reasoning"]
                text, token_ids, boundary = native_gold_packet(tokenizer, context, target, tools,
                                                               prompt_rendering="full")
                identifier = f"{row['id']}:action:{index:04d}"
                if identifier in seen_ids:
                    raise ValueError(f"duplicate derived teacher turn identity: {identifier}")
                seen_ids.add(identifier)
                row_data = {"id": identifier, "split": "train", "source_groups": sorted(set(program["source_groups"])),
                            "tokenizer_sha256": source_fingerprint, "text": text,
                            "token_ids": token_ids, "supervised_suffix_start": boundary,
                            "source_twin_id": identifier, "source_record_ids": [row["id"], identifier]}
                graph = text_row_graph(row_data, source_tokenizer_fingerprint=source_fingerprint,
                    tokenizer=tokenizer, expanded_tokenizer_fingerprint=expanded_fingerprint,
                    context_weight=.05)
                graph["conversion"]["teacher_source"] = {
                    "schema": "natlang.static-teacher-turn-source/1",
                    "kind": "static-transformation",
                    "interpretation": "ordinary-text-full-context-turn; no inferred native communication edge",
                    "source_artifact": str(args.source), "source_artifact_sha256": binding["source_sha256"],
                    "source_row_id": row["id"], "source_row_sha256": source_row_sha,
                    "source_program_id": program["id"],
                    "source_groups": sorted(set(program["source_groups"])),
                    "source_record_sha256": (row.get("provenance") or {}).get("program_ir_sha256"),
                    "teacher_response_artifact": str(original_path),
                    "teacher_response_sha256": original["sha256"],
                    "collection_receipt": binding['collections'][str(Path(original['source']).parents[2] / 'teacher-source.json')],
                    "selection_proof_sha256": binding["proof_sha256"]["selection"],
                    "global_protection_join_sha256": binding["proof_sha256"]["global_protection_join"],
                    "transfer_receipt_sha256": binding["proof_sha256"]["transfer"],
                    "teacher_model": (binding["teacher"].get("source") or {}).get("model_repo"),
                    "teacher_revision": (binding["teacher"].get("source") or {}).get("model_revision"),
                    "teacher_run_id": row.get("execution_identity", {}).get("execution_run_id"),
                    "teacher_turn_index": index, "teacher_invocation_id": turn.get("invocation_id"),
                    "request_sha256": turn.get("request_sha256"),
                    "raw_response_sha256": turn.get("raw_response_sha256"),
                    "source_transport": (row.get("provenance") or {}).get("text_neuralese_transport"),
                }
                writer.write(graph)
                teacher_turns += 1
        train = writer.finish()
        test = held.finish()
    except BaseException:
        writer.abort()
        held.abort()
        raise
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "natlang.verified-teacher-native-graph-build/1", "status": "passed",
        "model_key": args.model_key, "model_id": model_id,
        "source": {"path": str(args.source), "sha256": binding["source_sha256"],
                   "rows": len(row_hashes), "row_bindings": row_hashes},
        "proofs": {name: {"path": str(path), "sha256": binding["proof_sha256"][name]}
                   for name, path in binding["proof_paths"].items()},
        "tokenizer": {"source_fingerprint": source_fingerprint,
                      "expanded_fingerprint": expanded_fingerprint, "role_token_ids": list(role_ids)},
        "current_protection": binding["current"]["proof"],
        "turns": teacher_turns, "context_weight": .05,
        "outputs": {"train": train, "test": test},
        "conversion": "captured full prompt + exact teacher assistant action, ordinary one-node text graph; "
                      "captured text-marker stand-in is not an authentic Neuralese edge"}
    temp = args.output / ".receipt.json.tmp"
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, args.output / "receipt.json")
    print(json.dumps({"model_key": args.model_key, "model_id": model_id,
                      "turns": teacher_turns, "outputs": receipt["outputs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
