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


def current_protection(path, *, verify_current=True):
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
    live_review = source_review.REGISTRY / 'holds.jsonl'
    if verify_current and sha256_file(live_review) != manifest['source_review_sha256']:
        raise ValueError('protection projection differs from current source reviews')
    pending = source_review.pending_unpinned_source_groups()
    component_file = manifest.get('source_group_components')
    if component_file is not None:
        if component_file not in {item['path'] for item in manifest['files']}:
            raise ValueError('source component projection lacks its immutable file binding')
        from scripts.neuralese_data.splits import close_group_sets
        closed, _ = close_group_sets(
            (row['source_groups'] for _, row in file_rows(path.parent / component_file)),
            {'held': held, 'quality': quality, 'pending': pending})
        held, quality, pending = (closed[key] for key in ('held', 'quality', 'pending'))
    if len(held) != manifest['held_groups'] or len(quality) != manifest['quality_groups']:
        raise ValueError('current protection group projection is incomplete')
    return {'held': held, 'quality': quality, 'pending': pending,
            'proof': {'path': str(path), 'sha256': sha256_file(path),
                'held_groups': len(held), 'quality_groups': len(quality),
                'source_review_sha256': manifest['source_review_sha256']}}



def validate_authored_packet(paths, selection, packet_sha, programs, groups):
    """Verify task/reference witnesses; actual teacher decisions qualify separately."""
    required = {'reference_qualification', 'reference_summary', 'selection_protection'}
    if not required <= paths.keys():
        raise ValueError('fresh authored packet needs exact reference and protection witnesses')
    qualification = json.loads(paths['reference_qualification'].read_text())
    summary = json.loads(paths['reference_summary'].read_text())
    if (not programs or type(selection.get('cases')) is not int
            or type(selection.get('worlds')) is not int
            or qualification.get('schema') != 'natlang.authored-reference-qualification/1'
            or qualification.get('passed') is not True
            or qualification.get('training_admission') is not False
            or qualification.get('reference_traces_are_teacher_generations') is not False
            or qualification.get('cases_sha256') != packet_sha
            or qualification.get('selection_proof_sha256') != sha256_file(paths['selection'])
            or qualification.get('reference_summary_sha256') != sha256_file(paths['reference_summary'])
            or type(qualification.get('cases')) is not int or qualification['cases'] != len(programs)
            or qualification.get('counts') != summary or summary.get('cases') != len(programs)
            or any(type(summary.get(key)) is not int for key in ('cases', 'admitted', 'decisions', 'approved'))
            or summary.get('admitted') != len(programs)
            or summary.get('omitted') != {} or summary.get('omissions') != []
            or any(not isinstance(qualification.get(key+'_inventory'), dict)
                   or not qualification[key+'_inventory']
                   or canonical_json_sha256_hex(qualification[key+'_inventory']) != qualification.get(key+'_sha256')
                   for key in ('runtime', 'scripts'))
            or canonical_json_sha256_hex(programs) != selection.get('canonical_programs_sha256')
            or len(groups) != selection.get('worlds')):
        raise ValueError('fresh authored reference qualification differs from its complete original task packet')
    reference = selection.get('protection') or {}
    projection_path = paths['selection_protection']
    projection = current_protection(projection_path, verify_current=False)
    if (sha256_file(projection_path) != reference.get('sha256')
            or any(projection['proof'].get(key) != reference.get(key)
                   for key in ('source_review_sha256', 'held_groups', 'quality_groups'))
            or groups & (projection['held'] | projection['quality'])):
        raise ValueError('fresh authored packet historical protection binding or source groups differ')


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
    fresh_selection = selection.get('schema') == 'natlang.fresh-authored-teacher-packet/1'
    code_selection = selection.get('schema') == 'natlang.strong-teacher-code-source-selection/1'
    structured_selection = code_selection or fresh_selection
    packet_sha = selection.get('cases_sha256') if structured_selection else selection.get('output_sha256')
    if (sha256_file(paths['selected_cases']) != packet_sha or
            (structured_selection and (selection.get('held_and_pending_intersections') != 0
                                 or selection.get('training_admission') is not False)) or
            (not structured_selection and (selection.get('schema') != 'natlang.native-teacher-task-selection/1'
                                    or selection.get('status') != 'passed'))):
        raise ValueError('selected task packet differs from its passed immutable selection proof')
    if (protection.get('schema') != 'natlang.native-teacher-global-protection-join/1'
            or protection.get('status') != 'passed'
            or protection.get('selected_packet_sha256') != packet_sha
            or protection.get('held_group_intersections') or protection.get('quality_group_intersections')):
        raise ValueError('teacher selection intersects protected evaluation or quality-held groups')
    programs, packet_groups, ordered_programs = {}, set(), []
    for _, row in file_rows(paths['selected_cases']):
        program = row.get('program_ir', row)
        if not isinstance(program.get('id'), str) or program['id'] in programs:
            raise ValueError('selected task packet needs distinct program identities')
        if structured_selection and (program.get('version') != 'natlang.program/2' or program.get('split') != 'train'
                or not isinstance(program.get('source_groups'), list) or not program['source_groups']
                or any(not isinstance(group, str) or not group for group in program['source_groups'])):
            raise ValueError('selected code packet needs actual train program source groups')
        packet_groups.update(program.get('source_groups', []))
        programs[program['id']] = canonical_json_sha256_hex(program)
        ordered_programs.append(program)
    if structured_selection and selection.get('cases') != len(programs):
        raise ValueError('selected code packet count differs from selection proof')
    if not structured_selection and set(programs) != set(selection.get('selected_ids', [])):
        raise ValueError('selected packet program IDs differ from selection proof')
    train_groups = packet_groups if structured_selection else set(selection.get('train_groups', []))
    held_groups = set(selection.get('held_groups', []))
    if code_selection and 'selected_source_groups' in selection:
        declared_groups = selection['selected_source_groups']
        valid = ((type(declared_groups) is int and declared_groups == len(train_groups)) or
                 (isinstance(declared_groups, list) and set(declared_groups) == train_groups))
        if not valid:
            raise ValueError('selected code packet groups differ from selection proof')
    if fresh_selection:
        validate_authored_packet(paths, selection, packet_sha, ordered_programs, packet_groups)
    if train_groups & held_groups:
        raise ValueError('selection proof train/held source groups overlap')
    execution = prepared_execution_packet(paths, packet_sha, ordered_programs)
    if execution is not None:
        programs = {row['id']: canonical_json_sha256_hex(row) for row in execution['programs']}
        train_groups = {group for row in execution['programs'] for group in row.get('source_groups', [])}
    return {'selection': selection, 'packet_sha256': packet_sha, 'protection': protection, 'selected_ids': set(programs),
            'program_sha256': programs, 'train_groups': train_groups, 'held_groups': held_groups,
            **({'execution_preparation': execution['preparation']} if execution is not None else {}),
            'proofs': {name: {'path': str(path), 'sha256': sha256_file(path)} for name, path in paths.items()}}


def prepared_execution_packet(paths, source_sha256, source_programs):
    """Authenticate the existing rollout preparer's exact execution view."""
    names={'execution_packet','execution_preparation'}
    present=names & paths.keys()
    if not present:
        return None
    if present!=names:
        raise ValueError('prepared teacher execution requires both packet and preparation witnesses')
    preparation=json.loads(paths['execution_preparation'].read_text())
    if (preparation.get('schema')!='natlang.task-rollout-preparation/1'
            or preparation.get('source_sha256')!=source_sha256
            or preparation.get('packet_sha256')!=sha256_file(paths['execution_packet'])
            or preparation.get('root_mode') not in ('student','authored')
            or not isinstance(preparation.get('source_cases'),str)
            or not Path(preparation['source_cases']).is_absolute()):
        raise ValueError('teacher execution preparation differs from its original source or packet')
    from natlang_neuralese.rl.task_rollouts import student_root_row, authored_root_row
    transform=student_root_row if preparation['root_mode']=='student' else authored_root_row
    originals={row['id']:(index,row) for index,row in enumerate(source_programs)}
    executed=[];positions=[]
    for _,row in file_rows(paths['execution_packet']):
        selected=originals.get(row.get('id'))
        if selected is None or canonical_json_sha256_hex(row)!=canonical_json_sha256_hex(transform(selected[1])):
            raise ValueError('teacher execution packet differs from its declared root-mode transformation')
        positions.append(selected[0]);executed.append(row)
    if not positions or positions!=list(range(positions[0],positions[0]+len(positions))):
        raise ValueError('teacher execution packet must be an exact contiguous source slice')
    return {'programs':executed,'preparation':preparation}


def collection_provenance(collection, transferred, *, seen=()):
    """Authenticate continuation inputs without changing their actor or schema."""
    schema = collection.get('schema')
    if schema not in {'natlang.sandbox-strong-teacher-collection/1',
                       'natlang.sandbox-strong-teacher-continuation/1'}:
        raise ValueError('collection receipt differs from supported collection/continuation schemas')
    nested = collection.get('continuation')
    if nested is not None:
        if (schema != 'natlang.sandbox-strong-teacher-collection/1'
                or not isinstance(nested, dict)
                or nested.get('schema') != 'natlang.sandbox-strong-teacher-continuation/1'):
            raise ValueError('collection has an invalid nested teacher continuation')
        for name, value in nested.items():
            if name not in {'schema', 'at'} and name in collection and collection[name] != value:
                raise ValueError('nested teacher continuation conflicts with outer ' + name)
        # The immutable outer receipt supplies the restored packet contract. Keep
        # nested actor/runtime identities exactly as written, without rewriting it.
        collection = {**collection, **nested}
    if schema.endswith('collection/1') and 'original_source' not in collection:
        return {}
    digest = canonical_json_sha256_hex(collection)
    if digest in seen:
        raise ValueError('teacher continuation parent cycle')
    proofs = {}
    references = {name: collection.get(name) for name in (
        'original_source', 'original_queue', 'drain_frontier', 'actor',
        'actual_teacher_execution_identity')}
    history = collection.get('metadata_history', [])
    if not isinstance(history, list):
        raise ValueError('teacher continuation metadata history is not a list')
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise ValueError('teacher continuation metadata history lacks a binding')
        references['metadata_history:' + str(index)] = entry.get('binding')
    for name, reference in references.items():
        if (not isinstance(reference, dict) or not isinstance(reference.get('path'), str)
                or not isinstance(reference.get('sha256'), str) or len(reference['sha256']) != 64
                or type(reference.get('bytes')) is not int or reference['bytes'] < 1):
            raise ValueError('teacher continuation lacks its exact ' + name + ' binding')
        copies = transferred.get(reference['path'], [])
        if not copies:
            raise ValueError('teacher continuation provenance was not transferred: ' + name)
        for item in copies:
            local = Path(item['path'])
            if (item.get('sha256') != reference['sha256'] or item.get('bytes') != reference['bytes']
                    or local.stat().st_size != reference['bytes'] or sha256_file(local) != reference['sha256']):
                raise ValueError('teacher continuation provenance differs from its exact ' + name)
        proofs[name] = {'path': str(Path(copies[0]['path'])), 'sha256': reference['sha256'],
                        'bytes': reference['bytes'], 'source': reference['path']}
    execution = json.loads(Path(proofs['actual_teacher_execution_identity']['path']).read_text())
    if not isinstance(execution, dict) or execution.get('actor') != collection['actor']:
        raise ValueError('teacher continuation execution identity refers to a different actor')
    parent = json.loads(Path(proofs['original_source']['path']).read_text())
    if (parent.get('source_cases_sha256') != collection.get('source_cases_sha256')
            or parent.get('protected_selection') != collection.get('protected_selection')
            or collection.get('source_indices_seeds_preserved') is not True
            or parent.get('seed_base') != collection.get('seed_base')
            or type(collection.get('entries')) is not int or collection['entries'] < 1):
        raise ValueError('teacher continuation changed its original selected source or seed contract')
    for name, proof in collection_provenance(parent, transferred, seen=(*seen, digest)).items():
        proofs['parent:' + name] = proof
    return proofs


def restore_transfer_row_ids(transfer, additions):
    """Add missing transport identities without replacing existing facts."""
    identities={}
    row_ids=set()
    for item in additions:
        row_id=item.get('row_id')
        identity=tuple(item.get(key) for key in ('path','source','sha256','bytes'))
        if not isinstance(row_id,str) or not row_id or row_id in row_ids or identity in identities:
            raise ValueError('teacher response row-binding continuation has duplicate or invalid identities')
        row_ids.add(row_id);identities[identity]=row_id
    patched=[];used=set()
    for item in transfer.get('files',[]):
        identity=tuple(item.get(key) for key in ('path','source','sha256','bytes'))
        if identity not in identities:
            patched.append(item);continue
        row_id=identities[identity]
        if identity in used or (item.get('row_id') is not None and item['row_id']!=row_id):
            raise ValueError('teacher response row-binding continuation changes an existing identity')
        used.add(identity);patched.append({**item,'row_id':row_id})
    if used!=set(identities):
        raise ValueError('teacher response row-binding continuation contains an unbound source file')
    result={**transfer,'files':patched}
    mapped=[item['row_id'] for item in patched if isinstance(item.get('row_id'),str)]
    if len(mapped)!=len(set(mapped)):
        raise ValueError('teacher response transfer repeats a row identity')
    return result


def source_binding(args):
    if getattr(args, 'scaffold_harvest', None) is not None:
        return scaffold_source_binding(args)
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
    # A few early private transfers omitted row_id on their per-response file
    # records. Keep that historical receipt immutable and accept only an
    # explicit, byte-bound continuation which restores the missing mapping.
    row_bindings_path = root / 'source-row-bindings.json'
    if row_bindings_path.exists():
        row_bindings = json.loads(row_bindings_path.read_text())
        if (row_bindings.get('schema') != 'natlang.teacher-transfer-row-bindings/1'
                or row_bindings.get('parent_transfer_receipt_sha256') != sha256_file(proof_paths['transfer'])
                or row_bindings.get('combined_source_sha256') != actual_source_sha
                or not isinstance(row_bindings.get('files'), list)):
            raise ValueError('teacher response row-binding continuation does not match its exact parent receipt/source')
        transfer = restore_transfer_row_ids(transfer, row_bindings['files'])
        proof_paths['source_row_bindings'] = row_bindings_path
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
                'selection', 'selected_cases', 'global_protection_join',
                'reference_qualification', 'reference_summary', 'selection_protection',
                'execution_packet', 'execution_preparation') if key in batch})
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
    transferred = {}
    for item in transfer.get('files', []):
        transferred.setdefault(item.get('source'), []).append(item)
    for item in transfer.get('files', []):
        if not str(item.get('source', '')).endswith('/teacher-source.json'):
            continue
        if batch_map.exists() and item['source'] not in packets:
            continue  # A referenced original receipt is evidence, not another response batch.
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
        provenance = collection_provenance(collection, transferred)
        if (collection.get('source_cases_sha256') != packet['packet_sha256']
                or canonical_json_sha256_hex(collection.get('protected_selection')) != canonical_json_sha256_hex(packet['selection'])
                or canonical_json_sha256_hex(collection.get('teacher')) != canonical_json_sha256_hex(teacher)):
            raise ValueError('collection receipt differs from selected tasks or actual stronger teacher')
        prepared=packet.get('execution_preparation')
        if prepared is not None and (prepared['root_mode']!=collection.get('root_mode')
                or prepared['source_cases']!=collection.get('source_cases')):
            raise ValueError('collection receipt differs from its executed root mode or original source path')
        packets[item['source']] = packet
        collections[item['source']] = {'path': str(local), 'sha256': item['sha256'],
                                      'packet_proofs': packet['proofs'],
                                      **({'continuation_proofs': provenance} if provenance else {})}
        for name, proof in provenance.items():
            proof_paths['collection:' + item['source'] + ':' + name] = Path(proof['path'])
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


def scaffold_source_binding(args):
    """Bind the separately harvested 48-case scaffold batch without inventing a Qwen batch receipt.

    The harvest index is the exact source authority: it separates accepted and
    repair keys, binds every transferred byte, and points to each result
    manifest's original queue slot. We then feed those original trajectory
    rows through the same current protection and rendering path as other
    verified teacher data.
    """
    root = args.scaffold_harvest.resolve(strict=True)
    source = args.source.resolve(strict=True)
    harvest_path = root / 'harvest.json'
    ferry_path = root / 'ferry-receipt.json'
    harvest = json.loads(harvest_path.read_text())
    ferry = json.loads(ferry_path.read_text())
    accepted, repair = harvest.get('accepted_keys'), harvest.get('repair_keys')
    if (harvest.get('schema') != 'natlang.teacher-scaffold-qualification-harvest/1'
            or not isinstance(accepted, list) or not isinstance(repair, list)
            or ferry.get('verified') is not True
            or ferry.get('accepted') != len(accepted)
            or ferry.get('repair') != len(repair) or ferry.get('sha256') is None
            or sha256_file(root / 'harvest.tar.gz') != ferry['sha256']
            or (root / 'harvest.tar.gz').stat().st_size != ferry.get('bytes')):
        raise ValueError('scaffold harvest archive or terminal ferry receipt is not exact')
    files = harvest.get('files')
    if not isinstance(files, dict) or len(files) != ferry.get('verified_files'):
        raise ValueError('scaffold harvest file inventory differs from its terminal ferry receipt')
    for relative, binding in files.items():
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or not path.is_file() or type(binding.get('bytes')) is not int:
            raise ValueError('scaffold harvest contains an unsafe or incomplete file binding')
        if path.stat().st_size != binding['bytes'] or sha256_file(path) != binding.get('sha256'):
            raise ValueError('scaffold harvest artifact differs from its immutable inventory: ' + relative)
    if (len(accepted) < 1 or len(set(accepted)) != len(accepted)
            or len(set(repair)) != len(repair)
            or set(accepted) & set(repair) or set(accepted) | set(repair) != {
                str(Path(name).parts[1]) for name in files if name.startswith(('accepted/', 'repair-private-writer-return/'))}):
        raise ValueError('scaffold accepted/repair inventory does not partition all returned task keys')

    metadata = root / 'metadata'
    cases_path, queue_path = metadata / 'qualification-cases.jsonl', metadata / 'queue.jsonl'
    cases = [row for _, row in file_rows(cases_path)]
    queue = [row for _, row in file_rows(queue_path)]
    if len(cases) != 12 or len(queue) != 48:
        raise ValueError('scaffold source packet/queue is not the complete 12-task/48-case batch')
    case_by_id = {row['id']: row for row in cases}
    queue_by_key = {row.get('key'): row for row in queue}
    if len(case_by_id) != 12 or len(queue_by_key) != 48 or set(queue_by_key) != set(accepted) | set(repair):
        raise ValueError('scaffold queue does not uniquely cover every accepted and repair key')
    identity_path = metadata / 'teacher-execution-identity.json'
    actor_path = metadata / 'teacher-serving-actor.json'
    execution_identity, actor = json.loads(identity_path.read_text()), json.loads(actor_path.read_text())
    actor_ref = execution_identity.get('actor') or {}
    if (actor.get('schema') != 'natlang.self-hosted-teacher-actor/1'
            or actor.get('teacher') != 'nvidia/Qwen3.8-Flash-Next-FP4-Local'
            or actor.get('revision') != 'b9ce18a6887e18c089fdea8509a33e63429a3300'
            or actor_ref.get('sha256') != sha256_file(actor_path)
            or actor_ref.get('bytes') != actor_path.stat().st_size
            or execution_identity.get('model_id') != 'pennyroyal'):
        raise ValueError('scaffold teacher actor and execution identity do not authenticate one another')
    launch_path = metadata / 'launch.json'
    if not launch_path.exists():
        launch_path = metadata / 'launch-nodepath-retry.json'
    launch = json.loads(launch_path.read_text())
    queue_binding = launch.get('queue') or {}
    source_binding = launch.get('source') or {}
    argv = launch.get('argv')
    source_mismatch = bool(source_binding) and (
        source_binding.get('sha256') != sha256_file(cases_path)
        or source_binding.get('rows') != len(cases))
    if (launch.get('actor', {}).get('sha256') != sha256_file(identity_path)
            or queue_binding.get('sha256') != sha256_file(queue_path)
            or queue_binding.get('rows') != len(queue)
            or source_mismatch or not isinstance(argv, list) or not argv):
        raise ValueError('scaffold actual launch does not bind this actor and complete queue')

    # The prepared JSONL is only a byte-exact concatenation of accepted
    # originals. Every row remains independently tied to its original result
    # file and collector manifest below.
    source_rows = [row for _, row in file_rows(source)]
    if len(source_rows) != len(accepted) or len({row.get('id') for row in source_rows}) != len(accepted):
        raise ValueError('scaffold combined source must contain every accepted unique full trajectory')
    row_bindings, source_files = [], {}
    selected_ids, train_groups, packet_programs = set(), set(), {}
    collections_key = 'scaffold-harvest:' + sha256_file(harvest_path)
    for key in accepted:
        task = queue_by_key[key]
        program_id = task.get('source_program_id')
        program = case_by_id.get(program_id)
        if program is None or program.get('split') != 'train' or program.get('version') != 'natlang.program/2':
            raise ValueError('scaffold queue points outside its exact train task packet')
        result_path = root / 'accepted' / key / 'results.jsonl'
        manifest_path = result_path.with_name('results.jsonl.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        index = task['index']
        result_rows = [row for _, row in file_rows(result_path)]
        if (manifest.get('version') != 'natlang.teacher_batch.native/1'
                or manifest.get('source_sha256') != sha256_file(cases_path)
                or manifest.get('range') != {'start': index, 'count': 1}
                or manifest.get('completed') != 1 or manifest.get('missing')
                or manifest.get('output_sha256') != sha256_file(result_path)
                or len(result_rows) != 1):
            raise ValueError('accepted scaffold result lacks its exact completed queue-slot receipt: ' + key)
        row = result_rows[0]
        if row.get('id') != source_rows[len(row_bindings)].get('id'):
            raise ValueError('combined scaffold source order differs from exact accepted-key order')
        if (row.get('task', {}).get('program_ir') != program
                or row.get('provenance', {}).get('program_ir_sha256') != canonical_json_sha256_hex(program)):
            raise ValueError('accepted trajectory task differs from the exact original train task')
        selected_ids.add(program_id)
        packet_programs[program_id] = canonical_json_sha256_hex(program)
        groups = program.get('source_groups')
        if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
            raise ValueError('scaffold program lacks unique source groups')
        train_groups.update(groups)
        source_files[row['id']] = {'row_id': row['id'], 'path': str(result_path),
            'sha256': sha256_file(result_path), 'bytes': result_path.stat().st_size,
            'source': str(result_path), 'collection_source': collections_key}
        row_bindings.append({'key': key, 'id': row['id'], 'program_id': program_id,
            'result': {'path': str(result_path), 'sha256': sha256_file(result_path),
                       'bytes': result_path.stat().st_size},
            'manifest': {'path': str(manifest_path), 'sha256': sha256_file(manifest_path),
                         'bytes': manifest_path.stat().st_size}})
    if {row.get('id') for row in source_rows} != {item['id'] for item in row_bindings}:
        raise ValueError('combined scaffold source changed row membership')
    current = current_protection(args.protection_manifest)
    if train_groups & (current['held'] | current['quality'] | current['pending']):
        raise ValueError('scaffold train groups intersect current protection')
    selection = {'schema': 'natlang.teacher-scaffold-accepted-selection/1',
        'accepted_keys': accepted, 'repair_keys_excluded': repair,
        'selected_ids': sorted(selected_ids), 'train_groups': sorted(train_groups),
        'qualification_cases_sha256': sha256_file(cases_path), 'queue_sha256': sha256_file(queue_path),
        'training_admission': 'pending independent source and current-protection validation'}
    packet = {'selection': selection, 'packet_sha256': sha256_file(cases_path),
        'protection': {'schema': 'natlang.scaffold-current-protection-join/1', 'status': 'passed',
            'accepted_keys': len(accepted), 'held_intersections': 0, 'quality_intersections': 0},
        'selected_ids': selected_ids, 'program_sha256': packet_programs,
        'train_groups': train_groups, 'held_groups': set(),
        'proofs': {'selection': {'path': str(harvest_path), 'sha256': sha256_file(harvest_path)},
                   'global_protection_join': {'path': str(args.protection_manifest),
                                              'sha256': current['proof']['sha256']}}}
    proofs = {'harvest': harvest_path, 'ferry': ferry_path, 'transfer': ferry_path, 'source': source,
        'qualification_cases': cases_path, 'queue': queue_path,
        'teacher_execution_identity': identity_path, 'teacher_actor': actor_path,
        'teacher_configuration_basis': metadata / 'teacher-configuration-basis.json',
        'launch': launch_path}
    for item in row_bindings:
        proofs['accepted:' + item['key'] + ':result'] = Path(item['result']['path'])
        proofs['accepted:' + item['key'] + ':manifest'] = Path(item['manifest']['path'])
    proof_sha = {name: sha256_file(path) for name, path in proofs.items()}
    return {'source_sha256': sha256_file(source), 'transfer': {'schema': 'natlang.scaffold-accepted-source-binding/1',
            'training_admission': False, 'combined': {'path': str(source), 'sha256': sha256_file(source),
            'bytes': source.stat().st_size}, 'files': list(source_files.values())},
        'teacher': {'schema': actor.get('schema'), 'teacher': actor['teacher'], 'revision': actor['revision'],
                    'actor_path': str(actor_path), 'actor_sha256': sha256_file(actor_path),
                    'execution_identity_path': str(identity_path),
                    'execution_identity_sha256': sha256_file(identity_path)},
        'evidence': {}, 'source_files': source_files, 'selected_ids': selected_ids,
        'train_groups': train_groups, 'held_groups': set(), 'proof_paths': proofs,
        'proof_sha256': proof_sha, 'collections': {collections_key: {'path': str(harvest_path),
            'sha256': sha256_file(harvest_path), 'accepted_rows': len(row_bindings),
            'repair_rows_excluded': len(repair)}}, 'packets': {collections_key: packet},
        'current': current, 'scaffold': {'row_bindings': row_bindings, 'selection': selection}}


def prepare_scaffold_source(source: Path, harvest_root: Path):
    """Create one deterministic JSONL stream from the accepted original result rows."""
    root = harvest_root.resolve(strict=True)
    harvest = json.loads((root / 'harvest.json').read_text())
    if harvest.get('schema') != 'natlang.teacher-scaffold-qualification-harvest/1':
        raise ValueError('unsupported scaffold harvest')
    if source.exists():
        raise ValueError('refusing to overwrite an existing scaffold source stream')
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = source.with_name(source.name + '.tmp')
    try:
        with temporary.open('xb') as output:
            for key in harvest.get('accepted_keys', []):
                result = root / 'accepted' / key / 'results.jsonl'
                rows = list(file_rows(result))
                if len(rows) != 1:
                    raise ValueError('accepted scaffold key must contain one whole trajectory: ' + key)
                output.write(rows[0][0].rstrip(b'\r\n') + b'\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, source)
        directory = os.open(source.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def teacher_collection_source(file_receipt, binding):
    """Join an exact transferred response to its verified collection receipt."""
    if 'collection_source' in file_receipt:
        source = file_receipt['collection_source']
        if not isinstance(source, str) or not source:
            raise ValueError('teacher response collection_source must be an exact source path')
    else:
        source = str(Path(file_receipt['source']).parents[2] / 'teacher-source.json')
    if source not in binding['collections'] or source not in binding['packets']:
        raise ValueError('teacher response is not bound to its immutable collection batch')
    return source


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
    batch_source = teacher_collection_source(file_receipt, binding)
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
    if len(originals) != 1:
        raise ValueError('combined teacher trajectory differs from its original response: ' + row['id'])
    if row.get('source_quality_continuation'):
        from scripts.neuralese_data.teacher_file_quality import verify_quality_continuation
        verify_quality_continuation(row, originals[0], original)
    elif canonical_json_sha256_hex(originals[0]) != canonical_json_sha256_hex(row):
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
    # Missing capture identity is an explicit omission, never a legacy equality
    # join in which two absent identifiers qualify a positive target.
    invocation = turn.get('invocation_id')
    hashes = [turn.get(name) for name in ('request_sha256', 'raw_response_sha256')]
    if (not isinstance(invocation, str) or not invocation.strip() or any(
            not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)
            for value in hashes)):
        return False, {**decision, 'reason': 'original provider turn lacks a nonempty invocation and lowercase SHA-256 request/response capture'}
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


def teacher_action_graph_id(source_record_id: str, source_record_sha256: str, turn_index: int,
                            rendering_profile: str | None = None) -> str:
    """Bind a rendered action to its exact trajectory, not only its reusable task ID.

    Teacher batches can legitimately execute the same selected program more than
    once with different captured contexts and answers.  The program/action key
    alone therefore is not a graph identity.  The exact source-record digest is
    stable across tokenizer views and conversions while distinguishing those
    separate whole trajectories.
    """
    if (not isinstance(source_record_id, str) or not source_record_id
            or not isinstance(source_record_sha256, str) or len(source_record_sha256) != 64
            or any(char not in '0123456789abcdef' for char in source_record_sha256)
            or type(turn_index) is not int or turn_index < 0
            or rendering_profile not in (None, 'keep', 'last_turn_only')):
        raise ValueError('teacher action identity requires a source record ID, lowercase SHA-256, and nonnegative turn index')
    identifier = f'{source_record_id}:source-row:{source_record_sha256}:action:{turn_index:04d}'
    return (identifier if rendering_profile is None else
            identifier + ':history-reasoning:' + rendering_profile)


def unresolved_neuralese_references(value):
    """Return exact typed block IDs in a captured prompt/target, preserving nested content."""
    if isinstance(value, dict):
        if value.get('type') == 'neuralese' and isinstance(value.get('id'), str):
            yield value['id']
        for child in value.values():
            yield from unresolved_neuralese_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from unresolved_neuralese_references(child)


def unresolved_gold_error(error):
    """Classify only the two explicit unresolved-state failures from native_gold_packet."""
    message = str(error)
    if message == 'ordinary gold text prompt contains unresolved neuralese blocks':
        return 'prompt'
    if message == 'ordinary gold text target contains unresolved neuralese blocks':
        return 'target'
    return None


def _count_value(value, needle):
    if isinstance(value, str):
        return value.count(needle)
    if isinstance(value, dict):
        return sum(_count_value(child, needle) for child in value.values())
    if isinstance(value, list):
        return sum(_count_value(child, needle) for child in value)
    return 0


def captured_text_emulation_context(turn, context, tools):
    """Recover the exact provider-visible text context for authenticated text-emulated blocks."""
    references = list(unresolved_neuralese_references(context))
    if not references:
        return context, tools, None, None
    response = turn.get('model_response') or {}
    transport = response.get('transport_provenance') or {}
    omissions = {'unresolved_scope': 'prompt', 'unresolved_block_ids': sorted(set(references))}
    if (transport.get('version') != 'text-marker-standin/2' or transport.get('learned_vectors') is not False
            or transport.get('qualification_certificate') is not False or transport.get('training_admission') is not False
            or any(not isinstance(transport.get(name), str) or len(transport[name]) != 64
                   or any(ch not in '0123456789abcdef' for ch in transport[name])
                   for name in ('raw_request_sha256', 'rendered_request_sha256'))):
        return None, None, None, {**omissions,
            'reason': 'typed prompt blocks lack authenticated non-learned text-emulation transport evidence'}
    expanded = transport.get('expanded_input_blocks')
    if not isinstance(expanded, list):
        return None, None, None, {**omissions,
            'reason': 'typed prompt blocks lack captured expanded text-emulation bodies'}
    by_id = {}
    for block in expanded:
        block_type = block.get('type') if isinstance(block, dict) else None
        if (not isinstance(block, dict) or not isinstance(block.get('id'), str)
                or block['id'] in by_id or (block_type is not None and
                    (not isinstance(block_type, str) or not block_type.startswith('Neuralese<')))
                or not isinstance(block.get('body'), str) or block.get('learned_vectors') is not False):
            return None, None, None, {**omissions,
                'reason': 'captured text-emulation block metadata is incomplete or ambiguous'}
        body_sha = hashlib.sha256(block['body'].encode('utf-8')).hexdigest()
        if block.get('body_sha256') != body_sha:
            return None, None, None, {**omissions,
                'reason': 'captured text-emulation block body digest mismatch'}
        by_id[block['id']] = block
    if not set(references) <= set(by_id):
        return None, None, None, {**omissions,
            'reason': 'captured text-emulation bodies do not cover every typed context reference'}
    rendered = transport.get('rendered_messages')
    if not isinstance(rendered, list) or list(unresolved_neuralese_references(rendered)):
        return None, None, None, {**omissions,
            'reason': 'captured provider-visible messages are missing or still contain unresolved blocks'}
    for block_id in set(references):
        block = by_id[block_id]
        type_label = f' type={block["type"]}' if block.get('type') else ''
        label = (f'[[Neuralese text block id={block_id}{type_label}; exact JSON string body='
                 f'{json.dumps(block["body"], ensure_ascii=False, separators=(",", ":"))}]]')
        if _count_value(rendered, label) < references.count(block_id):
            return None, None, None, {**omissions,
                'reason': 'captured provider-visible messages do not contain the exact authenticated block expansion'}
    chosen_tools = tools
    template = transport.get('text_template_readout')
    if template is not None:
        if (not isinstance(template, dict) or template.get('schema') != 'natlang.text-template-readout/1'
                or template.get('learned_vectors') is not False
                or template.get('qualification_certificate') is not False
                or template.get('training_admission') is not False
                or template.get('call') != 'return_result' or template.get('value') != 'decode'):
            return None, None, None, {**omissions,
                'reason': 'captured typed-read template is not the declared non-learned text-emulation contract'}
        read_block = by_id.get(template.get('read_body_id'))
        if (read_block is None or read_block.get('body_sha256') != template.get('read_source_sha256')
                or read_block.get('type') != 'Neuralese<(v: Neuralese<unknown>) => unknown>'):
            return None, None, None, {**omissions,
                'reason': 'captured text-read instruction is not bound to its exact declared read body'}
        matches = [tool for tool in tools if (tool.get('function') or {}).get('name') == 'return_result']
        if len(matches) != 1:
            return None, None, None, {**omissions,
                'reason': 'typed text-read request lacks one exact return_result tool schema'}
        chosen_tools = matches
    proof = {'schema': 'natlang.captured-text-neuralese-input/1',
        'source': 'provider-captured-text-marker-standin', 'learned_vectors': False,
        'qualification_certificate': False, 'training_admission': False,
        'raw_request_sha256': transport['raw_request_sha256'],
        'rendered_request_sha256': transport['rendered_request_sha256'],
        'expanded_blocks': [{'id': block_id, 'type': by_id[block_id]['type'],
            'body_sha256': by_id[block_id]['body_sha256'], 'occurrences': references.count(block_id)}
            for block_id in sorted(set(references))]}
    return rendered, chosen_tools, proof, None


def render_teacher_packet(tokenizer, context, target, tools, turn=None):
    """Render one action or return an explicit omission when its exact state payload is unavailable."""
    emulation_proof = None
    if turn is not None:
        context, tools, emulation_proof, omission = captured_text_emulation_context(turn, context, tools)
        if omission is not None:
            return None, omission
    try:
        packet = native_gold_packet(tokenizer, context, target, tools, prompt_rendering='full')
        return (packet, emulation_proof), None
    except ValueError as error:
        scope = unresolved_gold_error(error)
        if scope is None:
            raise
        omitted_value = context if scope == 'prompt' else target
        block_ids = sorted(set(unresolved_neuralese_references(omitted_value)))
        if not block_ids:
            raise ValueError('unresolved neuralese error had no exact source block IDs') from error
        return None, {'reason': 'full ordinary-text conversion requires unresolved typed Neuralese payloads; '
                              'retain this action for source-bound native graph conversion',
                      'unresolved_scope': scope, 'unresolved_block_ids': block_ids}


def _content_index_policy_sha256():
    from scripts.neuralese_data.continue_native_source_indexes import POLICY_FILES
    package = Path(__file__).resolve().parents[2] / 'training/neuralese/natlang_neuralese'
    return canonical_json_sha256_hex({
        'converter': sha256_file(Path(__file__)),
        'parser': {name: sha256_file(package / name)
                   for name in (*POLICY_FILES, 'data/native_graph.py')}})


def seed_content_fingerprints(paths, fingerprint):
    from natlang_neuralese.common import row_index
    policy = _content_index_policy_sha256()
    seen, inputs = {}, []
    for path in paths:
        digest = sha256_file(path)
        key = {'source_sha256': digest, 'tokenizer_fingerprint': fingerprint,
               'policy_sha256': policy, 'kind': 'teacher-content-fingerprints-v1'}
        cache_path = row_index.entry_path('teacher-content', canonical_json_sha256_hex(key))
        cached = row_index.load(cache_path)
        contents = cached.get('fingerprints') if isinstance(cached, dict) else None
        count = cached.get('rows') if isinstance(cached, dict) else None
        valid = (isinstance(cached, dict) and cached.get('key') == key
            and type(count) is int and count >= 0 and isinstance(contents, dict)
            and count >= len(contents)
            and all(isinstance(content, str) and len(content) == 64
                    and all(char in '0123456789abcdef' for char in content)
                    and isinstance(identifier, str) and identifier
                    for content, identifier in contents.items())
            and cached.get('metadata_sha256') == canonical_json_sha256_hex(
                {'rows': count, 'fingerprints': contents}))
        if not valid:
            contents, count = {}, 0
            with open_jsonl_text(path) as source:
                for line in source:
                    if not line.strip():
                        continue
                    graph = json.loads(line)
                    parsed = parse_graph(graph, tokenizer_fingerprint=fingerprint)
                    if parsed.split != 'train':
                        raise ValueError('teacher deduplication references must be train graphs')
                    contents.setdefault(graph_content_sha256(graph), parsed.id)
                    count += 1
        if sha256_file(path) != digest:
            raise ValueError('teacher deduplication source changed during validation')
        if not valid:
            metadata = {'rows': count, 'fingerprints': contents}
            row_index.save(cache_path, {'key': key, **metadata,
                'metadata_sha256': canonical_json_sha256_hex(metadata)})
        for content, identifier in contents.items():
            seen.setdefault(content, {'path': str(path), 'id': identifier})
        inputs.append({'path': str(path), 'sha256': digest, 'rows': count})
    return seen, inputs


def explicit_history_profile_receipt(args, tokenizer):
    """Bind an explicit override without conflating it with a backbone default.

    The tokenizer's effective policy is always recorded under ``tokenizer``;
    this separate conditional field records who selected a nondefault
    rendering profile so downstream teacher admission can reconstruct IDs.
    """
    requested = getattr(args, 'history_reasoning_policy', None)
    actual = (getattr(tokenizer, 'natlang_history_reasoning', None) or {}).get('policy')
    if requested is not None and actual != requested:
        raise ValueError('explicit teacher rendering profile differs from the bound tokenizer profile')
    return ({'explicit_history_reasoning_policy': requested}
            if requested is not None else {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument('--scaffold-harvest', type=Path,
                        help='authenticate a separately harvested scaffold batch; never treats it as a standard Qwen batch')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument('--history-reasoning-policy', choices=('keep', 'last_turn_only'),
                        help='explicit rendering policy; omitted preserves the backbone default')
    parser.add_argument("--protection-manifest", type=Path, required=True)
    parser.add_argument('--deduplicate-against', type=Path, action='append', default=[])
    parser.add_argument("--model-key", choices=("lfm25", "mellum21"), required=True)
    args = parser.parse_args()
    if args.scaffold_harvest is not None and not args.source.exists():
        prepare_scaffold_source(args.source, args.scaffold_harvest)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output: {args.output}")
    binding = source_binding(args)
    decisions, decision_proof = teacher_decision_admissions(args.source, binding['source_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_tokenizer), trust_remote_code=False,
                                               local_files_only=True)
    from natlang_neuralese.serve.chat import bind_history_reasoning
    bind_history_reasoning(tokenizer, policy=args.history_reasoning_policy, require_declared=True,
                           allow_transition=args.history_reasoning_policy is not None)
    explicit_profile_receipt = explicit_history_profile_receipt(args, tokenizer)
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
            collection_source = teacher_collection_source(original, binding)
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
                packet, omission = render_teacher_packet(tokenizer, context, target, tools, turn)
                if omission is not None:
                    action_omissions.append({'source_row_id': row['id'], 'teacher_turn_index': index,
                        **omission, 'decision': decision})
                    continue
                (text, token_ids, boundary), emulation_proof = packet
                identifier = teacher_action_graph_id(row['id'], source_row_sha, index,
                    rendering_profile=getattr(args, 'history_reasoning_policy', None))
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
                    "collection_receipt": binding['collections'][collection_source],
                    "selection_proof_sha256": binding['packets'][collection_source]['proofs']['selection']['sha256'],
                    "global_protection_join_sha256": binding['packets'][collection_source]['proofs']['global_protection_join']['sha256'],
                    "transfer_receipt_sha256": binding["proof_sha256"]["transfer"],
                    "teacher_model": ((binding["teacher"].get("source") or {}).get("model_repo")
                                      or binding["teacher"].get('teacher')),
                    "teacher_revision": ((binding["teacher"].get("source") or {}).get("model_revision")
                                         or binding["teacher"].get('revision')),
                    "teacher_run_id": row.get("execution_identity", {}).get("execution_run_id"),
                    "teacher_turn_index": index, "teacher_invocation_id": turn.get("invocation_id"),
                    'decision_admission': decision,
                    **({'captured_text_emulation_input': emulation_proof} if emulation_proof else {}),
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
        **explicit_profile_receipt,
        "model_key": args.model_key, "model_id": model_id,
        "source": {"path": str(args.source), "sha256": binding["source_sha256"],
                   "rows": len(row_hashes), "row_bindings": row_hashes},
        "proofs": {name: {"path": str(path), "sha256": binding["proof_sha256"][name]}
                   for name, path in binding["proof_paths"].items()},
        "tokenizer": {"source_fingerprint": source_fingerprint,
                      "expanded_fingerprint": expanded_fingerprint, "role_token_ids": list(role_ids),
                      "history_reasoning": tokenizer.natlang_history_reasoning},
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
