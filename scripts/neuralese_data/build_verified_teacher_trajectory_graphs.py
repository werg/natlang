#!/usr/bin/env python3
"""Convert selected, exact-success teacher traces into ordinary native text graphs.

This static transformation preserves each captured request context and the
teacher's actual next assistant turn. Text-emulated Neuralese remains ordinary
text here; this tool does not invent producer/consumer edges. Such edges need
the separate source-authenticated semantic-pair converter.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.neuralese_data.build_native_semantic_pairs_e2e import CompressedRows, provision
from scripts import source_review

from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_graph import parse_graph
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


def packet_binding(root, references):
    """Bind a batch to its actual immutable packet, independent of future feeds."""
    paths = {}
    for name, reference in references.items():
        path = Path(reference['path'])
        if not path.is_absolute():
            path = root / path
        if sha256_file(path) != reference['sha256']:
            raise ValueError('batch packet witness differs from its immutable SHA: ' + name)
        paths[name] = path
    selection = json.loads(paths['selection'].read_text())
    protection = json.loads(paths['global_protection_join'].read_text())
    code_selection = selection.get('schema') == 'natlang.strong-teacher-code-source-selection/1'
    packet_sha = selection.get('cases_sha256') if code_selection else selection.get('output_sha256')
    if (sha256_file(paths['selected_cases']) != packet_sha or
            (code_selection and (selection.get('held_and_pending_intersections') != 0
                                 or selection.get('training_admission') is not False)) or
            (not code_selection and (selection.get('schema') != 'natlang.native-teacher-task-selection/1'
                                    or selection.get('status') != 'passed'))):
        raise ValueError('selected task packet differs from its passed immutable selection proof')
    if (protection.get('schema') != 'natlang.native-teacher-global-protection-join/1'
            or protection.get('status') != 'passed'
            or protection.get('selected_packet_sha256') != packet_sha
            or protection.get('held_group_intersections') or protection.get('quality_group_intersections')):
        raise ValueError('teacher selection intersects protected evaluation or quality-held groups')
    programs, packet_groups = {}, set()
    for _, row in file_rows(paths['selected_cases']):
        program = row.get('program_ir', row)
        if not isinstance(program.get('id'), str) or program['id'] in programs:
            raise ValueError('selected task packet needs distinct program identities')
        if code_selection and (program.get('version') != 'natlang.program/2' or program.get('split') != 'train'
                or not isinstance(program.get('source_groups'), list) or not program['source_groups']
                or any(not isinstance(group, str) or not group for group in program['source_groups'])):
            raise ValueError('selected code packet needs actual train program source groups')
        packet_groups.update(program.get('source_groups', []))
        programs[program['id']] = canonical_json_sha256_hex(program)
    if code_selection and selection.get('cases') != len(programs):
        raise ValueError('selected code packet count differs from selection proof')
    if not code_selection and set(programs) != set(selection.get('selected_ids', [])):
        raise ValueError('selected packet program IDs differ from selection proof')
    train_groups = packet_groups if code_selection else set(selection.get('train_groups', []))
    held_groups = set(selection.get('held_groups', []))
    if code_selection and 'selected_source_groups' in selection:
        declared_groups = selection['selected_source_groups']
        valid = ((type(declared_groups) is int and declared_groups == len(train_groups)) or
                 (isinstance(declared_groups, list) and set(declared_groups) == train_groups))
        if not valid:
            raise ValueError('selected code packet groups differ from selection proof')
    if train_groups & held_groups:
        raise ValueError('selection proof train/held source groups overlap')
    return {'selection': selection, 'packet_sha256': packet_sha, 'protection': protection, 'selected_ids': set(programs),
            'program_sha256': programs, 'train_groups': train_groups, 'held_groups': held_groups,
            'proofs': {name: {'path': str(path), 'sha256': sha256_file(path)} for name, path in paths.items()}}


def source_binding(args):
    root = args.source.parent
    batch_map = root / 'batch-source-bindings.json'
    proof_paths = {name: root / filename for name, filename in {
        'transfer': 'transfer-receipt.json', 'evidence_source': 'evidence-source-v2.json',
        'teacher_runtime': 'penny-server-binding.json'}.items()
        if name == 'transfer' or not batch_map.exists() or (root / filename).exists()}
    proofs = {name: json.loads(path.read_text()) for name, path in proof_paths.items()}
    actual_source_sha = sha256_file(args.source)
    transfer, teacher = proofs['transfer'], proofs.get('teacher_runtime')
    if (transfer.get('schema') != 'natlang.verified-teacher-trajectory-transfer/1'
            or transfer.get('combined', {}).get('path') != str(args.source)
            or transfer.get('combined', {}).get('sha256') != actual_source_sha
            or transfer.get('combined', {}).get('bytes') != args.source.stat().st_size
            or transfer.get('training_admission') is not False):
        raise ValueError('transfer receipt does not bind the exact raw teacher trajectory source')
    if teacher is not None and teacher.get('teacher_role') != 'independent stronger open teacher; not either student/onpolicy':
        raise ValueError('teacher provenance does not identify an independent teacher')
    packets = {}
    if batch_map.exists():
        proof_paths['batch_source_bindings'] = batch_map
        mapping = json.loads(batch_map.read_text())
        if mapping.get('schema') != 'natlang.teacher-batch-source-bindings/1' or not mapping.get('batches'):
            raise ValueError('teacher snapshot needs explicit batch source bindings')
        for batch in mapping['batches']:
            source = batch['collection_source']
            if source in packets:
                raise ValueError('teacher snapshot repeats a batch source binding')
            packets[source] = packet_binding(root, {key: batch[key] for key in (
                'selection', 'selected_cases', 'global_protection_join')})
        fallback = None
    else:
        fallback_refs = {name: {'path': str(root / filename), 'sha256': sha256_file(root / filename)}
            for name, filename in {'selection': 'selection-proof.json', 'selected_cases': 'selected-cases.jsonl',
                                   'global_protection_join': 'global-protection-join.json'}.items()}
        fallback = packet_binding(root, fallback_refs)
        proof_paths.update({name: Path(value['path']) for name, value in fallback_refs.items()})
    source_files = {item.get('row_id'): item for item in transfer.get('files', [])
                    if isinstance(item.get('row_id'), str)}
    collections = {}
    for item in transfer.get('files', []):
        if not str(item.get('source', '')).endswith('/teacher-source.json'):
            continue
        local = Path(item['path'])
        if sha256_file(local) != item['sha256'] or local.stat().st_size != item['bytes']:
            raise ValueError('original collection receipt differs from its transfer binding')
        packet = packets.get(item['source'], fallback)
        if packet is None:
            raise ValueError('teacher collection needs exactly one immutable packet binding')
        previous = collections.get(item['source'])
        if previous is not None and previous['sha256'] != item['sha256']:
            raise ValueError('teacher collection has conflicting duplicate receipt bindings')
        collection = json.loads(local.read_text())
        if teacher is None:
            teacher = collection.get('teacher')
        if (not isinstance(teacher, dict) or teacher.get('teacher_role') !=
                'independent stronger open teacher; not either student/onpolicy'):
            raise ValueError('collection does not identify an independent stronger teacher')
        if (collection.get('schema') != 'natlang.sandbox-strong-teacher-collection/1'
                or collection.get('source_cases_sha256') != packet['packet_sha256']
                or canonical_json_sha256_hex(collection.get('protected_selection')) != canonical_json_sha256_hex(packet['selection'])
                or canonical_json_sha256_hex(collection.get('teacher')) != canonical_json_sha256_hex(teacher)):
            raise ValueError('collection receipt differs from selected tasks or actual stronger teacher')
        packets[item['source']] = packet
        collections[item['source']] = {'path': str(local), 'sha256': item['sha256'],
                                      'packet_proofs': packet['proofs']}
    if not collections or set(packets) != set(collections):
        raise ValueError('teacher trajectories need complete immutable per-batch collection receipts')
    proof_sha = {name: sha256_file(path) for name, path in proof_paths.items()}
    return {'source_sha256': actual_source_sha, 'transfer': transfer, 'teacher': teacher,
            'evidence': proofs.get('evidence_source', {}), 'source_files': source_files,
            'selected_ids': set().union(*(value['selected_ids'] for value in packets.values())),
            'train_groups': set().union(*(value['train_groups'] for value in packets.values())),
            'held_groups': set().union(*(value['held_groups'] for value in packets.values())),
            'proof_paths': proof_paths, 'proof_sha256': proof_sha, 'collections': collections,
            'packets': packets, 'current': current_protection(args.protection_manifest)}


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
    packet = binding['packets'][batch_source]
    if (program['id'] not in packet['selected_ids']
            or canonical_json_sha256_hex(program) != packet['program_sha256'][program['id']]
            or not set(groups) <= packet['train_groups'] or set(groups) & packet['held_groups']):
        raise ValueError('teacher program differs from its actual batch selected packet')
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


def teacher_decision_admissions(source, source_sha256):
    """Reuse the native materializer, once per exact source/runtime for both ports."""
    if sha256_file(source) != source_sha256:
        raise ValueError('teacher decision source differs from its immutable SHA-256')
    repo = Path(__file__).resolve().parents[2]
    cli = repo / 'ts-host/scripts/materialize-native-teacher.mjs'
    assets = [cli, repo / 'ts-host/scripts/jsonl-stream.mjs',
              *sorted((repo / 'ts-host/dist').rglob('*.js'))]
    if not any(path.name == 'native-materializer.js' for path in assets):
        raise ValueError('build the native teacher materializer before converting teacher graphs')
    runtime = {str(path.relative_to(repo)): sha256_file(path) for path in assets}
    identity = {'source_sha256': source_sha256, 'runtime': runtime, 'direct_answers': True}
    key = canonical_json_sha256_hex(identity)
    cache = Path(source).parent / 'decision-admission-cache'
    cache.mkdir(exist_ok=True)
    destination = cache / key
    with (cache / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix='.materializing-', dir=cache) as temporary:
                staged = Path(temporary)
                output = staged / 'decisions.jsonl'
                completed = subprocess.run(['node', str(cli), str(Path(source).resolve()), str(output),
                                            '--direct-answers'], capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError('native teacher materializer failed: ' + completed.stderr[-4000:])
                if sha256_file(source) != source_sha256 or any(
                        sha256_file(repo / path) != digest for path, digest in runtime.items()):
                    raise ValueError('teacher source or materializer runtime changed during admission')
                receipt = {'schema': 'natlang.verified-teacher-decision-materialization/1',
                    'identity': identity, 'output_sha256': sha256_file(output),
                    'summary': json.loads(completed.stdout.strip().splitlines()[-1])}
                receipt['summary']['output'] = 'decisions.jsonl'
                write_json_durable(staged / 'receipt.json', receipt)
                staged.rename(destination)
        receipt_path = destination / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        output = destination / 'decisions.jsonl'
        if (receipt.get('schema') != 'natlang.verified-teacher-decision-materialization/1'
                or receipt.get('identity') != identity
                or sha256_file(output) != receipt.get('output_sha256')):
            raise ValueError('teacher decision materialization differs from its exact source/runtime binding')
    decisions = {}
    for line in output.open():
        record = json.loads(line)
        reference, decision = record.get('source_ref') or {}, record.get('decision') or {}
        item = (reference.get('trajectory_id'), decision.get('index'))
        if not isinstance(item[0], str) or type(item[1]) is not int or item in decisions:
            raise ValueError('materialized teacher decisions lack unique original turn identities')
        decisions[item] = {'approved': (record.get('training_admission') or {}).get('approved') is True
                and decision.get('training_approved') is True and decision.get('failed_action') is False,
            'reason': (record.get('training_admission') or {}).get('reason'),
            'invocation_id': reference.get('invocation_id'),
            'raw_response_sha256': decision.get('source_raw_response_sha256'),
            'materialized_decision_id': record['id'],
            'materialized_decision_sha256': canonical_json_sha256_hex(record)}
    return decisions, {'path': str(receipt_path), 'sha256': sha256_file(receipt_path),
                       'output_sha256': receipt['output_sha256'], 'summary': receipt['summary']}


def admitted_teacher_turn(row, index, turn, decisions):
    """Join existing admission to the unchanged captured original action."""
    decision = decisions.get((row['id'], index))
    if decision is None:
        return False, {'reason': 'shared native materializer did not qualify this original decision'}
    if (decision['invocation_id'] != turn.get('invocation_id')
            or decision['raw_response_sha256'] != turn.get('raw_response_sha256')):
        raise ValueError('teacher decision admission disagrees with the original invocation/response')
    if not decision['approved']:
        return False, decision
    assistant = turn.get('assistant') or {}
    if not (turn.get('model_response') or {}).get('raw_calls') and not str(assistant.get('content') or '').strip():
        return False, {**decision, 'reason': 'no original provider action or nonempty assistant reply'}
    return True, decision


def graph_content_sha256(graph):
    return canonical_json_sha256_hex({'tokenizer_fingerprint': graph['tokenizer_fingerprint'],
                                     'nodes': graph['nodes']})


def seed_content_fingerprints(paths, fingerprint):
    seen, inputs = {}, []
    for path in paths:
        digest = sha256_file(path)
        count = 0
        with open_jsonl_text(path) as source:
            for line in source:
                if not line.strip():
                    continue
                graph = json.loads(line)
                parsed = parse_graph(graph, tokenizer_fingerprint=fingerprint)
                if parsed.split != 'train':
                    raise ValueError('teacher deduplication references must be train graphs')
                seen.setdefault(graph_content_sha256(graph), {'path': str(path), 'id': parsed.id})
                count += 1
        if sha256_file(path) != digest:
            raise ValueError('teacher deduplication source changed during validation')
        inputs.append({'path': str(path), 'sha256': digest, 'rows': count})
    return seen, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--protection-manifest", type=Path, required=True)
    parser.add_argument('--deduplicate-against', type=Path, action='append', default=[])
    parser.add_argument("--model-key", choices=("lfm25", "mellum21"), required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {args.output}")
    binding = source_binding(args)
    decisions, decision_proof = teacher_decision_admissions(args.source, binding['source_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_tokenizer), trust_remote_code=False,
                                               local_files_only=True)
    source_fingerprint = tokenizer_fingerprint(tokenizer)
    role_ids = provision(tokenizer)
    expanded_fingerprint = tokenizer_fingerprint(tokenizer)
    inventory = base_inventory(args.base_tokenizer)
    model_type = json.loads((args.base_tokenizer / "config.json").read_text())["model_type"]
    model_id = model_type + ":" + canonical_json_sha256_hex(inventory)
    seen_contents, dedup_inputs = seed_content_fingerprints(args.deduplicate_against, expanded_fingerprint)
    duplicate_omissions = []
    writer = GraphWriter(args.output / "train" / "native-graphs.jsonl.zst")
    held = GraphWriter(args.output / "test" / "native-graphs.jsonl.zst")
    seen_ids = set()
    row_hashes = []
    teacher_turns = 0
    action_omissions = []
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
                admitted, decision = admitted_teacher_turn(row, index, turn, decisions)
                if not admitted:
                    action_omissions.append({'source_row_id': row['id'], 'teacher_turn_index': index,
                        'reason': decision.get('reason'), 'decision': decision})
                    continue
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
                    "selection_proof_sha256": binding['packets'][str(Path(original['source']).parents[2] / 'teacher-source.json')]['proofs']['selection']['sha256'],
                    "global_protection_join_sha256": binding['packets'][str(Path(original['source']).parents[2] / 'teacher-source.json')]['proofs']['global_protection_join']['sha256'],
                    "transfer_receipt_sha256": binding["proof_sha256"]["transfer"],
                    "teacher_model": (binding["teacher"].get("source") or {}).get("model_repo"),
                    "teacher_revision": (binding["teacher"].get("source") or {}).get("model_revision"),
                    "teacher_run_id": row.get("execution_identity", {}).get("execution_run_id"),
                    "teacher_turn_index": index, "teacher_invocation_id": turn.get("invocation_id"),
                    'decision_admission': decision,
                    'decision_materialization_receipt_sha256': decision_proof['sha256'],
                    "request_sha256": turn.get("request_sha256"),
                    "raw_response_sha256": turn.get("raw_response_sha256"),
                    "source_transport": (row.get("provenance") or {}).get("text_neuralese_transport"),
                }
                teacher_turns += 1
                content = graph_content_sha256(graph)
                if content in seen_contents:
                    duplicate_omissions.append({'id': graph['id'], 'content_sha256': content,
                        'reason': 'exact token IDs and target weights already represented',
                        'represented_by': seen_contents[content]})
                    continue
                seen_contents[content] = {'path': str(writer.path), 'id': graph['id']}
                writer.write(graph)
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
        'decision_admission': decision_proof, 'action_omissions': action_omissions,
        'deduplication': {'inputs': dedup_inputs, 'omissions': duplicate_omissions,
            'omitted_rows': len(duplicate_omissions), 'emitted_rows': train['rows']},
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
