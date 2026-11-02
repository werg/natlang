#!/usr/bin/env python3
"""Stage a manifest append using authenticated production-parser leaf indexes.

This prepares data and geometry only.  It does not admit the source, change a
learner, or claim a behavioral qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_bytes_durable, write_json_durable
from natlang_neuralese.train import native_packed_capacity as capacity
from scripts.neuralese_data.build_acquired_streaming_successor import (
    PREPARED_SCHEMA, appended_manifest, bind, checked, read,
    receipt_protection_digest, require_output_headroom,
)

CODE = Path(__file__).read_bytes()


def prepare(predecessor_path, combined_path, census_path, output, *, reviewed_census_builder_sha256=None):
    predecessor_path, combined_path, census_path = [Path(p).resolve(strict=True)
        for p in (predecessor_path, combined_path, census_path)]
    pred_ref, combined_ref, census_ref = map(bind, (predecessor_path, combined_path, census_path))
    previous, combined, census = map(read, (predecessor_path, combined_path, census_path))
    if (combined.get('schema') != 'natlang.combined-acquired-screened-delta/1'
            or combined.get('training_admission') is not False
            or combined.get('predecessor', {}).get('sha256') != pred_ref['sha256']
            or census.get('schema') != 'natlang.acquired-effective-learning-census/1'
            or census.get('training_admission') is not False
            or census.get('screen', {}).get('sha256') != combined_ref['sha256']
            or census.get('rows') != combined.get('rows')
            or census.get('counts', {}).get('zero_effective_rows') != 0
            or census.get('ce_policy', {}).get('sha256') != previous['ce_policy']['sha256']):
        raise ValueError('manifest append lacks its exact predecessor and completed positive-loss census')
    source_path = checked(previous['source'], 'predecessor source')
    old_map = read(checked(previous['family_map'], 'predecessor family map'))
    prior_geometry_path = checked(previous['geometry'], 'predecessor geometry')
    graph_path = checked(combined['source'], 'combined source manifest')
    graph_ref = {**bind(graph_path), 'rows': combined['rows'],
                 'tokenizer_fingerprint': combined['tokenizer_fingerprint']}
    graph = read(graph_path)
    geometry = read(prior_geometry_path)
    tokenizer, roles = geometry['identity']['tokenizer'], geometry['identity']['roles']
    if (graph.get('schema') != 'natlang.native-graph-sources/1'
            or len(graph.get('sources', [])) != combined['source_leaves']
            or tokenizer != graph_ref['tokenizer_fingerprint']
            or old_map.get('schema') != 'natlang.logical-family-map/1'
            or old_map.get('source_sha256') != previous['source']['sha256']
            or len(old_map.get('rows', [])) != previous['rows']):
        raise ValueError('manifest, predecessor map or tokenizer changed')
    protection = ProtectedWorlds.load()
    if (protection.digest != combined['protected_worlds_sha256']
            or protection.digest != receipt_protection_digest(previous)):
        raise ValueError('protected world union changed')
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
    from scripts.neuralese_data.decision_holds import load_decision_holds
    quality_path = Path(combined['quality_protection']['path'])
    hold_root = Path(__file__).resolve().parents[2] / 'training/decision-holds'
    hold_paths = sorted(hold_root.glob('*.jsonl'))
    if (current_protection(quality_path)['proof'] != combined['quality_protection']
            or load_decision_holds(hold_paths)[1] != combined['decision_holds']):
        raise ValueError('quality protection or decision holds changed')
    checked(census['row_signals'], 'completed row census')
    checked(census['content_review'], 'completed content review')
    names = ('prepare_acquired_manifest_index_census.py', 'parallel_native_graph_index.py',
             'acquired_content_review.py', 'audit_acquired_learning_signal.py',
             'build_acquired_streaming_successor.py')
    components = census.get('code_bindings') or []
    if len(components) != len(names): raise ValueError('census code inventory is incomplete')
    for name, component in zip(names, components, strict=True):
        captured = checked(component, 'captured index/census source')
        current_sha = bind(Path(__file__).with_name(name))['sha256']
        historical_ok = (name == 'build_acquired_streaming_successor.py' and
                         reviewed_census_builder_sha256 == component['sha256'])
        if captured.name != name or (current_sha != component['sha256'] and not historical_ok):
            raise ValueError('index/census source is not the reviewed implementation')
    checked(census['pending_census'], 'original pending source census')
    seen_screens, leaf_inventory = set(), []
    screens, overlaps = combined.get('screens') or [], combined.get('occurrence_proofs') or []
    if not screens or len(screens) != len(overlaps):
        raise ValueError('combined screen and overlap inventory is incomplete')
    for screen_ref, overlap_ref in zip(screens, overlaps, strict=True):
        screen = read(checked(screen_ref, 'source protection screen'))
        overlap = read(checked(overlap_ref, 'source occurrence overlap'))
        if (screen.get('schema') != 'natlang.new-acquired-source-screen/1'
                or screen.get('training_admission') is not False
                or screen.get('prepared_source_sha256') != previous['source']['sha256']
                or screen.get('protected_worlds_sha256') != protection.digest
                or screen.get('quality_protection') != combined['quality_protection']
                or screen.get('decision_holds') != combined['decision_holds']
                or overlap.get('schema') != 'natlang.acquired-source-occurrence-overlap/1'
                or overlap.get('training_admission') is not False
                or overlap.get('screen') != screen_ref or overlap.get('predecessor') != combined['predecessor']
                or overlap.get('active_rows') != screen['active']['rows']
                or overlap.get('source_occurrence_suffix_missing') != 0
                or overlap.get('duplicate_source_occurrences_within_active') != 0
                or overlap.get('predecessor_source_occurrence_overlaps')
                or any(overlap.get('reserved_source_occurrence_overlaps',{}).values())
                or {ref['sha256'] for ref in overlap.get('reserved_screens',[])} != seen_screens
                or screen.get('pending',{}).get('rows') != 0
                or screen.get('counts') != {'active':screen['active']['rows'], 'raw':screen['active']['rows']}):
            raise ValueError('combined source overlap or screen evidence differs')
        leaf_inventory.extend(screen.get('output_inventory') or [])
        seen_screens.add(screen_ref['sha256'])
    if ([{k: item[k] for k in ('path','sha256','bytes')} for item in leaf_inventory] !=
            [{k: item[k] for k in ('path','sha256','bytes')} for item in graph['sources']]):
        raise ValueError('combined source leaves differ from screened inventory')
    histories = census.get('source_format_history') or []
    if len(histories) != len(leaf_inventory): raise ValueError('source history review is incomplete')
    for history, leaf in zip(histories, leaf_inventory, strict=True):
        if history.get('source') != {k:leaf[k] for k in ('path','sha256','bytes')}:
            raise ValueError('source history review binds another leaf')
        converter = read(checked(history['converter'], 'reviewed source converter receipt'))
        checked(history['export'], 'reviewed normalized source export')
        if (leaf not in converter.get('outputs', []) or converter.get('training_admission') is not False
                or converter.get('explicit_history_reasoning_policy') != history['history_reasoning_policy']
                or converter.get('export_receipt') != history['export']):
            raise ValueError('source conversion or history profile changed')
        if history.get('original_converter'): checked(history['original_converter'], 'original source converter')
    for obligation in (combined.get('pending_render_obligations') or {}).get('evidence', []):
        checked(obligation, 'retained render obligation')
    if combined.get('carried_over_context_obligations'):
        checked(combined['carried_over_context_obligations']['partition'], 'retained context obligation')
    proof_refs = census.get('leaf_index_proofs') or []
    if len(proof_refs) != len(graph['sources']):
        raise ValueError('leaf parser proof inventory is incomplete')
    proofs = []
    for leaf, reference in zip(graph['sources'], proof_refs, strict=True):
        proof = read(checked(reference, 'completed leaf parser proof'))
        if (proof.get('schema') != 'natlang.parallel-native-source-index/1'
                or proof.get('source') != {k: leaf[k] for k in ('path', 'sha256', 'bytes')}
                or proof.get('decision_readouts') != 0
                or proof.get('parser_policy_sha256') != geometry['identity']['parser_policy_sha256']):
            raise ValueError('leaf production parser evidence differs')
        proofs.append(proof)
    slots = census.get('ordered_kv_slots')
    if (not isinstance(slots, list) or len(slots) != combined['rows']
            or any(type(value) is not int or value < 1 for value in slots)):
        raise ValueError('combined capacity census lacks complete measured geometry')
    output = Path(output).resolve()
    if output.exists(): raise ValueError('source stage requires a fresh output directory')
    require_output_headroom(output)
    output.mkdir(parents=True)
    new_source_path = output / 'train-sources.json'
    write_json_durable(new_source_path, appended_manifest(source_path, graph_path, graph_ref))
    new_source, _ = capacity.index_source(new_source_path, tokenizer, roles)
    if (len(new_source.entries) != previous['rows'] + combined['rows']
            or [(row['id'], row['binding']) for row in new_source.entries[:previous['rows']]] !=
               [(row['id'], row['binding']) for row in old_map['rows']]):
        raise ValueError('manifest append changed the predecessor prefix or row count')
    additions = new_source.entries[previous['rows']:]
    if canonical_json_sha256_hex([row['id'] for row in additions]) != census['ordered_graph_ids_sha256']:
        raise ValueError('manifest parser order differs from the completed loss census')
    offset = 0
    for proof in proofs:
        end = offset + proof['rows']; rows = additions[offset:end]
        metadata = [{k: v for k, v in entry.items() if k != 'training_priority'} for entry in rows]
        if (canonical_json_sha256_hex(metadata) != proof['ordered_metadata_sha256']
                or canonical_json_sha256_hex(slots[offset:end]) != proof['ordered_geometry_sha256']):
            raise ValueError('manifest row bindings or geometry differ from their production leaf parser')
        offset = end
    if offset != len(additions): raise ValueError('leaf parser row totals differ')
    observed = {(row['id'], row['binding']): value for row, value in zip(additions, slots, strict=True)}
    _, geometry_ref, scan = capacity.geometry_for_source(new_source, output, prior=previous['geometry'],
        allow_build=True, indexed_geometry=observed)
    if (bind(predecessor_path) != pred_ref or bind(combined_path) != combined_ref
            or bind(census_path) != census_ref or bind(graph_path)['sha256'] != graph_ref['sha256']
            or ProtectedWorlds.load().digest != protection.digest
            or current_protection(quality_path)['proof'] != combined['quality_protection']
            or load_decision_holds(hold_paths)[1] != combined['decision_holds']
            or Path(__file__).read_bytes() != CODE):
        raise ValueError('source preparation controls changed during indexing')
    builder = output / ('builder-' + hashlib.sha256(CODE).hexdigest() + '.py')
    write_bytes_durable(builder, CODE)
    result = {'schema': PREPARED_SCHEMA, 'status': 'source and geometry prepared; no admission',
        'training_admission': False, 'predecessor': pred_ref, 'screen': combined_ref,
        'predecessor_source': bind(source_path), 'source': bind(new_source_path), 'geometry': geometry_ref,
        'graph': {k: graph_ref[k] for k in ('path','sha256','bytes')}, 'rows': len(new_source.entries),
        'new_rows': combined['rows'], 'tokenizer': tokenizer, 'roles': roles,
        'protection_digest': protection.digest, 'geometry_scan': scan, 'parallel_index': None,
        'combined_leaf_index_proofs': proof_refs, 'effective_loss_census': census_ref,
        'reviewed_census_builder_sha256': reviewed_census_builder_sha256,
        'pending_render_obligations': combined['pending_render_obligations'],
        'carried_over_context_obligations': combined['carried_over_context_obligations'],
        'builder': bind(builder)}
    write_json_durable(output / 'receipt.json', result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor-receipt', required=True)
    p.add_argument('--combined-receipt', required=True)
    p.add_argument('--census-receipt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reviewed-census-builder-sha256')
    a=p.parse_args()
    result=prepare(a.predecessor_receipt,a.combined_receipt,a.census_receipt,a.out,
        reviewed_census_builder_sha256=a.reviewed_census_builder_sha256)
    print(json.dumps({'rows':result['rows'],'new_rows':result['new_rows'],'training_admission':False}))


if __name__ == '__main__': main()
