#!/usr/bin/env python3
"""Stage an exact source/CE/geometry append from a completed acquired loss census.

Only bounded per-row metadata is retained. Graphs are indexed by the existing
streaming source validator; the raw corpus and census remain immutable. This
prepares a continuation, not learner publication, oracle verification or lift.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(os.environ.get('NATLANG_REPO') or Path(__file__).resolve().parents[2]).resolve()
sys.path.insert(0, str(ROOT / 'training/neuralese'))
sys.path.insert(0, str(ROOT))

from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from natlang_neuralese.common.jsonio import write_json_durable, write_bytes_durable
from natlang_neuralese.train import native_packed_capacity as capacity
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.own_output_supervision import continuation_identity, identity, policy_objective, prepare_policy
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, _data_continuation
from natlang_neuralese.data.native_sources import NativeGraphSource
from scripts.neuralese_data.build_logical_family_objective import capped_scales

SCHEMA = 'natlang.acquired-streaming-source-successor/1'
FAMILY = 'general-agent'
PREPARED_SCHEMA = 'natlang.acquired-streaming-source-preparation/1'
MODULE_SOURCE_BYTES = Path(__file__).read_bytes()


def sealed_builder(directory):
    digest = hashlib.sha256(MODULE_SOURCE_BYTES).hexdigest()
    target = Path(directory) / ('builder-' + digest + '.py')
    if target.exists():
        if target.read_bytes() != MODULE_SOURCE_BYTES: raise ValueError('sealed builder bytes differ')
    else: write_bytes_durable(target, MODULE_SOURCE_BYTES)
    return bind(target)


def checked_preparation_builder(prepared_path, reference):
    """An exact preserved source copy survives edits to its original checkout."""
    try: return bind(checked(reference, 'prepared source builder'))
    except (OSError, ValueError):
        target = Path(prepared_path).parent / ('builder-' + reference['sha256'] + '.py')
        return bind(checked({**reference, 'path': str(target)}, 'preserved prepared source builder'))


def bind(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path, trust_archive_receipt=False), 'bytes': path.stat().st_size}


def sft_capacity_descriptor(source, geometry):
    """Keep the executable descriptor within the runtime's strict wire schema."""
    return {'schema': 'natlang.native-sft-capacity-cases/1', 'source': source, 'geometry': geometry}


def checked(ref, name):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not isinstance(ref.get('sha256'), str):
        raise ValueError(f'{name} lacks an exact binding')
    actual = bind(ref['path'])
    if actual['sha256'] != ref['sha256'] or ('bytes' in ref and actual['bytes'] != ref['bytes']):
        raise ValueError(f'{name} changed')
    return Path(actual['path'])


def read(path):
    return json.loads(Path(path).read_bytes())


def file_signature(path):
    st = Path(path).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def require_output_headroom(path):
    path = Path(path).resolve()
    parent = path.parent
    if not parent.is_dir():
        raise ValueError('output parent must already exist on the intended data filesystem')
    if shutil.disk_usage(parent).free < 41 * 1024**3:
        raise ValueError('output filesystem cannot retain the 40 GiB floor and 1 GiB staging headroom')
    from natlang_neuralese.common.row_index import cache_dir
    cache = cache_dir().resolve()
    while not cache.exists() and cache != cache.parent:
        cache = cache.parent
    if shutil.disk_usage(cache).free < 41 * 1024**3:
        raise ValueError('row-index cache filesystem cannot retain the 40 GiB floor and 1 GiB staging headroom')


def appended_manifest(source_path, graph_path, graph_ref):
    source_path = Path(source_path).resolve(strict=True)
    manifest = read(source_path)
    if manifest.get('schema') != 'natlang.native-graph-sources/1' or not isinstance(manifest.get('sources'), list):
        raise ValueError('predecessor source manifest schema differs')
    # A copied manifest has a new parent directory. Preserve the exact physical
    # predecessor references when the original used relative paths.
    for item in manifest['sources']:
        reference = item.get('path')
        if not isinstance(reference, str) or not reference:
            raise ValueError('predecessor source path is absent')
        if not Path(reference).is_absolute():
            item['path'] = str((source_path.parent / reference).resolve(strict=True))
    manifest['sources'].append({'kind': 'native-graphs', 'path': str(graph_path), 'sha256': graph_ref['sha256'],
                                'bytes': graph_ref['bytes'], 'training_priority': 1})
    return manifest


def prepare_parallel_leaf(graph_path, graph_ref, tokenizer, roles, old_ids, output, *, workers):
    """Build the standard row cache using only the actual production parser in ordered workers."""
    from natlang_neuralese.common import row_index
    from natlang_neuralese.data import native_sources
    from scripts.neuralese_data.parallel_native_graph_index import parallel_rows
    code_path = ROOT / 'scripts/neuralese_data/parallel_native_graph_index.py'
    code_ref = bind(code_path)
    package = Path(native_sources.__file__).resolve().parents[1]
    parser_files = ('data/native_sources.py', 'data/native_graph_schema.py', 'data/decision_graph.py',
                    'model/native_graph_plan.py', 'data/native_pair_schema.py', 'common/row_index.py',
                    'common/jsonio.py', 'common/hashing.py')
    parser_code = {name: sha256_file_hex_cached(package / name) for name in parser_files}
    policy = canonical_json_sha256_hex(parser_code)
    key = {'source_sha256': graph_ref['sha256'], 'tokenizer': tokenizer, 'split': 'train',
           'policy_sha256': policy, 'kind': 'native-graphs', 'pair_model': None, 'role_ids': list(roles)}
    before = file_signature(graph_path)
    entries, observed, seen, decisions = [], {}, set(), 0
    for entry, slots, count in parallel_rows(graph_path, tokenizer, roles, workers=workers, max_pending=workers * 2):
        if entry['id'] in old_ids or entry['id'] in seen:
            raise ValueError('parallel source repeats a predecessor or new graph identity')
        seen.add(entry['id']); entries.append(entry); observed[(entry['id'], entry['binding'])] = slots
        decisions += count
        if len(entries) % 512 == 0:
            print(json.dumps({'kind': 'parallel_native_source_index', 'rows': len(entries),
                              'expected': graph_ref['rows'], 'workers': workers}), flush=True)
    if file_signature(graph_path) != before or len(entries) != graph_ref['rows'] or bind(code_path) != code_ref:
        raise ValueError('parallel source, builder or expected row count changed during indexing')
    if decisions:
        raise ValueError('acquired token-mass census omits finite decision readouts present in the source')
    cache = row_index.entry_path('native-graphs', canonical_json_sha256_hex(key))
    record = {'key': key, 'entries': entries, 'metadata_sha256': canonical_json_sha256_hex(entries)}
    existing = row_index.load(cache)
    if existing and (existing.get('key') != key or existing.get('entries') != entries
                     or existing.get('metadata_sha256') != record['metadata_sha256']):
        raise ValueError('existing native row index disagrees with independently parsed parallel rows')
    if not existing:
        row_index.save(cache, record)
    saved = row_index.load(cache)
    if not saved or any(saved.get(name) != value for name, value in record.items()):
        raise ValueError('native row index was not durably cached with exact ordered metadata')
    proof = {'schema': 'natlang.parallel-native-source-index/1', 'source': {'path': str(graph_path),
        'sha256': graph_ref['sha256'], 'bytes': graph_ref['bytes']}, 'rows': len(entries),
        'decision_readouts': decisions, 'parser_policy_sha256': policy, 'parser_code': parser_code,
        'ordered_metadata_sha256': record['metadata_sha256'],
        'ordered_geometry_sha256': canonical_json_sha256_hex([observed[(item['id'], item['binding'])] for item in entries]),
        'workers': workers, 'max_pending_raw_lines': workers * 2, 'parallel_builder': code_ref,
        'scope': 'actual production graph parser and row binding; no token arrays retained between rows'}
    proof_path = output / 'parallel-index-receipt.json'; write_json_durable(proof_path, proof)
    return observed, bind(proof_path)


def receipt_protection_digest(receipt):
    """Read the union binding, including already sealed successor receipts."""
    values = [receipt[key] for key in ('protection_digest', 'current_protection_digest') if key in receipt]
    if not values or any(value != values[0] for value in values):
        raise ValueError('source receipt has missing or conflicting protection union bindings')
    return values[0]


def predecessor_collision_count(screen, predecessor, old_rows):
    """Authenticate explicitly retained predecessor versions, never rename collisions."""
    if screen.get('pending_partition'):
        from scripts.neuralese_data.partition_acquired_context_pending import verify_partition
        original, _ = verify_partition(screen, predecessor)
        return predecessor_collision_count(original, predecessor, old_rows)
    reference = screen.get('predecessor_collision_filter')
    if reference is None:
        if screen.get('accounting', {}).get('predecessor_collisions', 0):
            raise ValueError('predecessor exclusions lack a collision filter proof')
        return 0
    proof = read(checked(reference, 'predecessor collision filter'))
    original = read(checked(proof['original_screen'], 'original protected screen'))
    old = {row['id']: row['binding'] for row in old_rows}
    excluded = proof.get('excluded')
    if (proof.get('schema') != 'natlang.acquired-predecessor-collision-filter/1'
            or proof.get('predecessor') != bind(predecessor)
            or proof.get('output_graph') != screen['output']
            or proof.get('input_graph') != original['output']
            or proof.get('protected_worlds_sha256') != screen['protected_worlds']['identity_set_sha256']
            or original.get('protected_worlds') != screen.get('protected_worlds')
            or original.get('coverage_audit') != screen.get('coverage_audit')
            or original.get('input_receipts') != screen.get('input_receipts')
            or original.get('excluded_current_protection') != screen.get('excluded_current_protection')
            or not isinstance(excluded, list)
            or original['output']['rows'] - len(excluded) != screen['output']['rows']
            or screen.get('accounting', {}).get('predecessor_collisions') != len(excluded)):
        raise ValueError('predecessor collision proof differs from source custody')
    checked(proof['builder'], 'collision filter builder')
    seen = set()
    for row in excluded:
        if (row.get('id') in seen or row.get('id') not in old
                or row.get('previous_binding') != old[row['id']]
                or not isinstance(row.get('candidate_binding'), str) or len(row['candidate_binding']) != 64
                or not isinstance(row.get('candidate_graph_sha256'), str) or len(row['candidate_graph_sha256']) != 64
                or row.get('equivalent_training_binding') is not (row['previous_binding'] == row['candidate_binding'])
                or row.get('disposition') != 'retain-qualified-predecessor; candidate retained in original immutable source'):
            raise ValueError('collision exclusion does not retain the actual predecessor binding')
        seen.add(row['id'])
    return len(excluded)


def prepare_source_stage(predecessor_receipt, screen_receipt, output, *, parallel_workers=2):
    """Index only the immutable successor and extend authenticated geometry before census completion."""
    predecessor_path, screen_path = (Path(p).resolve(strict=True) for p in (predecessor_receipt, screen_receipt))
    predecessor_identity, screen_identity = bind(predecessor_path), bind(screen_path)
    previous, screen = read(predecessor_path), read(screen_path)
    if screen.get('pending_partition'):
        from scripts.neuralese_data.partition_acquired_context_pending import verify_partition
        verify_partition(screen, predecessor_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('source preparation output must be a fresh immutable directory')
    require_output_headroom(output)
    source_path = checked(previous.get('source'), 'predecessor source')
    map_path = checked(previous.get('family_map'), 'predecessor family map')
    geometry_path = checked(previous.get('geometry'), 'predecessor geometry')
    protection = ProtectedWorlds.load()
    if (screen.get('schema') != 'natlang.acquired-current-protection-screen/1'
            or receipt_protection_digest(previous) != protection.digest
            or screen.get('protected_worlds', {}).get('identity_set_sha256') != protection.digest
            or screen.get('protected_worlds', {}).get('matches_expected') is not True):
        raise ValueError('source preparation does not bind the current complete protection union')
    graph_ref = screen['output']
    graph_path = Path(graph_ref['path']).resolve(strict=True)
    graph_before = file_signature(graph_path)
    if (type(graph_ref.get('rows')) is not int or graph_ref['rows'] <= 0
            or graph_before[2] != graph_ref.get('bytes')):
        raise ValueError('screened graph length or row count is invalid')
    # Fresh five-field checked digest; the source index separately validates
    # every graph and may use its authenticated row-index cache.
    if bind(graph_path)['sha256'] != graph_ref.get('sha256'):
        raise ValueError('screened graph bytes differ from the immutable screen')
    old_map = read(map_path)
    if (old_map.get('schema') != 'natlang.logical-family-map/1'
            or old_map.get('source_sha256') != previous['source']['sha256']
            or len(old_map.get('rows', [])) != previous['rows']):
        raise ValueError('predecessor family map is not source-bound')
    geometry = read(geometry_path)
    tokenizer, roles = geometry['identity']['tokenizer'], geometry['identity']['roles']
    if graph_ref.get('tokenizer_fingerprint') != tokenizer:
        raise ValueError('screen tokenizer differs from predecessor geometry')
    manifest = appended_manifest(source_path, graph_path, graph_ref)
    output.mkdir(parents=True)
    new_source_path = output / 'train-sources.json'; write_json_durable(new_source_path, manifest)
    parallel_proof = None
    if parallel_workers:
        old_ids = {item['id'] for item in old_map['rows']}
        if len(old_ids) != previous['rows']:
            raise ValueError('predecessor map repeats graph identities')
        observed, parallel_proof = prepare_parallel_leaf(graph_path, graph_ref, tokenizer, roles, old_ids, output, workers=parallel_workers)
        new_source, _ = capacity.index_source(new_source_path, tokenizer, roles)
        _, geometry_ref, scan = capacity.geometry_for_source(new_source, output, prior=previous['geometry'],
                                                             allow_build=True, indexed_geometry=observed)
    else:
        new_source, _, geometry_ref, scan = capacity.prepare_source(new_source_path, tokenizer, roles, output, prior=previous['geometry'])
    if (file_signature(graph_path) != graph_before or len(new_source.entries) != previous['rows'] + graph_ref['rows']
            or [(e['id'], e['binding']) for e in new_source.entries[:previous['rows']]] !=
               [(e['id'], e['binding']) for e in old_map['rows']]):
        raise ValueError('source preparation changed the predecessor prefix or screened graph file')
    if (bind(predecessor_path) != predecessor_identity or bind(screen_path) != screen_identity
            or checked(previous['source'], 'predecessor source') != source_path
            or checked(previous['family_map'], 'predecessor family map') != map_path
            or checked(previous['geometry'], 'predecessor geometry') != geometry_path
            or ProtectedWorlds.load().digest != protection.digest):
        raise ValueError('source preparation controls changed during indexing')
    result = {'schema': PREPARED_SCHEMA, 'status': 'source and geometry prepared; no loss census or admission',
        'training_admission': False, 'predecessor': bind(predecessor_path), 'screen': bind(screen_path),
        'predecessor_source': bind(source_path), 'source': bind(new_source_path), 'geometry': geometry_ref,
        'graph': {'path': str(graph_path), 'sha256': graph_ref['sha256'], 'bytes': graph_ref['bytes']},
        'rows': len(new_source.entries), 'new_rows': graph_ref['rows'], 'tokenizer': tokenizer, 'roles': roles,
        'protection_digest': protection.digest, 'geometry_scan': scan, 'parallel_index': parallel_proof,
        'builder': sealed_builder(output)}
    write_json_durable(output / 'receipt.json', result)
    return result


def scales_for_append(census, old_scales, new_mass, *, share=0.07, cap=8.0, family=FAMILY):
    """Keep prior effective proportions where the true cap permits, with mean-one token mass."""
    if not isinstance(family, str) or not re.fullmatch('[a-z][a-z0-9_-]*', family):
        raise ValueError('new logical family needs a valid explicit name')
    if not isinstance(share, (int, float)) or not 0 < share < 1 or not math.isfinite(share):
        raise ValueError('new family share must be finite and between zero and one')
    if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 1:
        raise ValueError('family cap must be finite and at least one')
    families = census.get('families') or {}
    if not set(families) <= set(old_scales) or (family in old_scales and family not in families):
        raise ValueError('predecessor census and declared family set differ')
    if type(new_mass) not in (int,float) or not math.isfinite(new_mass) or new_mass<=0:
        raise ValueError('new family contribution needs finite positive unscaled token mass')
    mass = {name: item.get('unscaled_mass') for name, item in families.items()}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in mass.values()):
        raise ValueError('every family needs finite positive unscaled token mass')
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= cap
           for value in old_scales.values()):
        raise ValueError('predecessor family scales are invalid')
    if not math.isclose(sum(mass[f] for f in families), census.get('unscaled_mass_total', -1), rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError('predecessor family mass total differs from census')
    previous_effective = {name: mass[name] * old_scales[name] for name in families}
    # Repeated admissions extend the same logical family. Preserve its old mass
    # and historical objective; allocate the requested share to old+new members.
    mass[family] = mass.get(family,0)+new_mass
    others={name:value for name,value in previous_effective.items() if name!=family}
    previous_total = sum(others.values())
    if previous_total<=0:raise ValueError('requested family share needs other existing families')
    targets = {name: (1 - share) * value / previous_total for name, value in others.items()}
    targets[family] = share
    scales, capped = capped_scales(targets, mass, cap)
    total = sum(mass.values())
    scaled = sum(mass[name] * scales[name] for name in mass)
    if (any(not math.isfinite(value) or not 0 < value <= cap for value in scales.values())
            or not math.isclose(scaled, total, rel_tol=1e-12, abs_tol=1e-6)):
        raise ValueError('bounded family scales do not have exact mean-one token mass')
    declared = {**{name: value for name, value in old_scales.items() if name not in families}, **scales}
    return {'scales': dict(sorted(declared.items())), 'unscaled_mass': mass, 'target_shares': targets,
            'effective_shares': {name: mass[name] * scales[name] / scaled for name in mass},
            'capped_families': capped, 'total_unscaled_mass': total, 'total_effective_mass': scaled,
            'previous_effective_mass': previous_effective, 'max_scale': cap}


def load_signals(path, *, census, screen, protected, old_ids):
    """Read compact census rows once; never retain graph payloads or token vectors."""
    rows, ids, mass, counts, stores = [], set(), Counter(), Counter(), Counter()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            ident = item.get('id')
            if not isinstance(ident, str) or not ident or ident in ids or ident in old_ids:
                raise ValueError('new census has a missing, repeated or predecessor graph ID')
            ids.add(ident)
            groups = item.get('source_groups')
            if (not isinstance(groups, list) or any(not isinstance(group, str) for group in groups)
                    or not isinstance(item.get('source_row_id'), str) or not item['source_row_id']
                    or not isinstance(item.get('source_store'), str) or not item['source_store']
                    or not isinstance(item.get('graph_sha256'), str) or len(item['graph_sha256']) != 64):
                raise ValueError('new census row lacks source, group or graph identity')
            category_mass = item.get('category_mass')
            if (not isinstance(category_mass, dict) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in category_mass.values()) or category_mass.get('context', 0) != 0
                    or category_mass.get('assistant_context', 0) != 0 or sum(category_mass.values()) <= 0
                    or item.get('eligible') is not True):
                raise ValueError('new census row lacks positive own-output supervision or exposes context loss')
            for key in ('forward_tokens', 'raw_positive_tokens', 'effective_positive_tokens'):
                if type(item.get(key)) is not int or item[key] < 1:
                    raise ValueError(f'new census row has invalid {key}')
                counts[key] += item[key]
            counts['source_label_diagnosis_rows'] += item.get('source_label_diagnosis') is True
            counts['oracle_verified_rows'] += item.get('oracle_verified') is True
            stores[item['source_store']] += 1
            mass.update(category_mass)
            case = {'id': ident, 'split': 'train', 'source_ids': [item['source_row_id']],
                    'source_groups': groups, 'split_groups': groups}
            if any(kind in {'id', 'group', 'binding'} for kind, _ in
                   world_identities(case, evidence_items=False) & protected.identities):
                raise ValueError('new census row intersects the current protection union')
            rows.append(item)
    expected_counts = census.get('counts') or {}
    if (len(rows) != census.get('rows') or len(rows) != screen['output']['rows']
            or expected_counts.get('zero_effective_rows') != 0
            or any(counts[key] != expected_counts.get(key, 0) for key in
                   ('forward_tokens', 'raw_positive_tokens', 'effective_positive_tokens',
                    'source_label_diagnosis_rows', 'oracle_verified_rows'))
            or dict(stores) != census.get('source_stores')
            or set(mass) != set(census.get('category_mass') or {})
            or any(not math.isclose(value, census['category_mass'][key], rel_tol=1e-12, abs_tol=1e-7)
                   for key, value in mass.items())):
        raise ValueError('census row inventory or effective mass differs from the completed receipt')
    return rows, dict(mass), dict(counts), dict(stores)


def verify_diagnosis_proof(rows, proof_path, program_path, screen_ref, ce_ref):
    diagnosis = {item['id']: item['source_store'] for item in rows if item.get('source_label_diagnosis') is True}
    if not diagnosis:
        if proof_path or program_path:
            raise ValueError('diagnosis proof was supplied for a source with no diagnosis rows')
        return None
    if not proof_path or not program_path:
        raise ValueError('diagnosis rows require their independently verified target proof and program')
    proof_path, program_path = Path(proof_path).resolve(strict=True), Path(program_path).resolve(strict=True)
    proof = read(proof_path)
    records = proof.get('rows')
    if (proof.get('schema') != 'natlang.acquired-source-diagnosis-target-verification/1'
            or proof.get('screen', {}).get('sha256') != screen_ref['sha256']
            or proof.get('ce_policy', {}).get('sha256') != ce_ref['sha256']
            or not isinstance(records, list) or proof.get('verified_rows') != len(records)
            or len(records) != len(diagnosis)):
        raise ValueError('diagnosis target proof does not bind the exact source and CE policy')
    seen = {}
    for item in records:
        ident, store, digest = item.get('id'), item.get('source_store'), item.get('target_sha256')
        if (ident in seen or ident not in diagnosis or diagnosis[ident] != store
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('diagnosis target proof has a missing, repeated or mismatched row')
        seen[ident] = store
    if seen != diagnosis:
        raise ValueError('diagnosis target proof does not cover the complete census subset')
    return {'receipt': bind(proof_path), 'program': bind(program_path), 'verified_rows': len(seen),
            'program_binding_scope': 'program supplied alongside receipt; the original receipt does not itself bind program bytes'}


def verify_conversion_custody(coverage, screen, screen_program):
    """Bind the exact converter and input-receipt inventory without rescanning graph payloads."""
    if not screen_program:
        raise ValueError('the source-screen program must be supplied for provenance review')
    screen_code = bind(screen_program)
    builder = bind(ROOT / 'scripts/neuralese_data/build_acquired_full_session_graphs.py')
    conversion = bind(ROOT / 'training/neuralese/natlang_neuralese/data/native_graph_conversion.py')
    graph_writer = bind(ROOT / 'scripts/neuralese_data/build_native_graph_text_corpus.py')
    inventory = []; historical_review = None; reviewed_receipts = 0
    for entry in coverage.get('receipts', []):
        receipt_path = checked({'path': entry['receipt'], 'sha256': entry['receipt_sha256']}, 'conversion receipt')
        receipt = read(receipt_path)
        files = (receipt.get('runtime') or {}).get('files') or {}
        accepted_builder, accepted_conversion = builder['sha256'], conversion['sha256']
        if (receipt.get('builder_sha256') != accepted_builder
                or files.get('native_graph_conversion', {}).get('sha256') != accepted_conversion):
            if historical_review is None:
                from scripts.neuralese_data.review_acquired_converter_equivalence import review
                historical_review = review(ROOT)
            if historical_review:
                allowed = lambda key: {ref['sha256'] for ref in historical_review['files'][key].values()}
                if (receipt.get('builder_sha256') in allowed('builder')
                        and files.get('native_graph_conversion', {}).get('sha256') in allowed('conversion')):
                    accepted_builder = receipt['builder_sha256']
                    accepted_conversion = files['native_graph_conversion']['sha256']
                    reviewed_receipts += 1
        if (receipt.get('schema') != 'natlang.acquired-full-session-graphs/1'
                or receipt.get('training_admission') is not False
                or receipt.get('builder_sha256') != accepted_builder
                or entry.get('builder_sha256') != accepted_builder
                or files.get('native_graph_conversion', {}).get('sha256') != accepted_conversion
                or files.get('graph_writer', {}).get('sha256') != graph_writer['sha256']
                or receipt.get('outputs') != entry.get('outputs') or receipt.get('counts') != entry.get('counts')
                or receipt.get('source_record_count') != entry.get('source_record_count')
                or receipt.get('export_receipt', {}).get('sha256') != entry.get('export_sha256')):
            raise ValueError('conversion receipt changed or does not bind the audited full-session graph converter')
        for output in receipt['outputs']:
            inventory.append({'path': str(receipt_path), 'sha256': entry['receipt_sha256'],
                'source_export': receipt['export_receipt'], 'source': receipt['source'], 'output': output})
    inventory.sort(key=lambda item: (item['source_export']['path'], item['output']['path']))
    if (len(inventory) != screen['input_receipts']['count']
            or canonical_json_sha256_hex(inventory) != screen['input_receipts']['sha256']):
        raise ValueError('screen input receipt inventory differs from its exact recorded digest')
    return {'converter': builder, 'graph_conversion': conversion, 'graph_writer': graph_writer, 'screen_program': screen_code,
        'screen_program_binding_scope': 'program supplied alongside receipt; the original screen receipt does not bind program bytes',
        'conversion_receipts': len(coverage['receipts']),
        'historical_converter_review': historical_review, 'historically_reviewed_receipts': reviewed_receipts,
        'screen_input_receipts_sha256': screen['input_receipts']['sha256'],
        'scope': 'bound converter emits one text node and no decision readouts; screen reports an input-only filter'}


def predecessor_mass_census(previous, predecessor_path, objective, *, seen=()):
    """Resolve measured mass or a checked append of measured contributions."""
    predecessor_path = Path(predecessor_path).resolve(strict=True)
    if predecessor_path in seen:
        raise ValueError('predecessor mass custody contains a cycle')
    unscaled = copy.deepcopy(objective); unscaled['family_balance'] = None
    if previous.get('objective_census') and previous.get('objective_census_receipt'):
        census_path = checked(previous['objective_census'], 'predecessor objective census')
        receipt_path = checked(previous['objective_census_receipt'], 'predecessor objective census receipt')
        census, receipt = read(census_path), read(receipt_path)
        policy = policy_objective(read(checked(census['policy'], 'predecessor census policy')))
        if (census.get('schema') != 'natlang.own-output-mass-census/1'
                or census.get('objective_without_family_scales') is not True
                or census.get('manifest', {}).get('sha256') != previous['source']['sha256']
                or census.get('rows') != previous['rows']
                or receipt.get('census', {}).get('sha256') != previous['objective_census']['sha256']
                or receipt.get('map', {}).get('sha256') != previous['family_map']['sha256']
                or policy != unscaled):
            raise ValueError('predecessor objective mass census does not bind its actual source and objective')
        return census, {'census': bind(census_path), 'receipt': bind(receipt_path)}
    if previous.get('schema') != SCHEMA:
        raise ValueError('predecessor lacks measured mass custody')
    parent_path = checked(previous['predecessor'], 'mass predecessor receipt')
    parent = read(parent_path)
    parent_objective = policy_objective(read(checked(parent['ce_policy'], 'mass predecessor policy')))
    parent_unscaled = copy.deepcopy(parent_objective); parent_unscaled['family_balance'] = None
    if parent_unscaled != unscaled:
        raise ValueError('unscaled loss objective changed across mass continuation')
    prior, prior_proof = predecessor_mass_census(parent, parent_path, parent_objective,
                                                seen=(*seen, predecessor_path))
    delta_path = checked(previous['effective_loss_census'], 'appended measured loss census')
    delta = read(delta_path)
    if (delta.get('schema') != 'natlang.acquired-effective-learning-census/1'
            or delta.get('screen') != previous.get('source_screen')
            or delta.get('ce_policy') != parent['ce_policy']
            or delta.get('objective_sha256') != canonical_json_sha256_hex(parent_objective)
            or delta.get('rows') != previous['new_rows']
            or parent['rows'] + delta['rows'] != previous['rows']
            or previous.get('predecessor_source') != parent['source']
            or delta.get('category_mass', {}).get('context', 0) != 0
            or delta.get('category_mass', {}).get('assistant_context', 0) != 0):
        raise ValueError('appended mass census does not bind the predecessor objective and source')
    mass = sum(delta['category_mass'].values())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('appended census has invalid measured loss mass')
    families = copy.deepcopy(prior['families']); name = previous['new_family']
    entry = families.setdefault(name, {'rows': 0, 'unscaled_mass': 0.})
    entry['rows'] += delta['rows']; entry['unscaled_mass'] += mass
    declared = previous['family_balance']['unscaled_mass']
    if (set(families) != set(declared)
            or any(not math.isclose(row['unscaled_mass'], declared[name], rel_tol=1e-10, abs_tol=1e-6)
                   for name, row in families.items())
            or sum(row['rows'] for row in families.values()) != previous['rows']):
        raise ValueError('declared family mass differs from measured append custody')
    total = sum(row['unscaled_mass'] for row in families.values())
    return {'schema': 'natlang.own-output-mass-census/1', 'manifest': previous['source'],
            'rows': previous['rows'], 'families': families, 'unscaled_mass_total': total,
            'objective_without_family_scales': True}, {
        'schema': 'natlang.measured-mass-append-custody/1', 'predecessor': bind(parent_path),
        'predecessor_mass': prior_proof, 'delta': bind(delta_path),
        'current_source': previous['source'], 'unscaled_objective': unscaled}


def build(predecessor_receipt, screen_receipt, census_receipt, output, *, share=0.07, prepared_source_receipt=None,
          diagnosis_verification=None, diagnosis_program=None, family=FAMILY, screen_program=None):
    predecessor_path, screen_path, census_path = map(lambda p: Path(p).resolve(strict=True),
        (predecessor_receipt, screen_receipt, census_receipt))
    previous, screen, census = map(read, (predecessor_path, screen_path, census_path))
    pending_count = 0
    if screen.get('pending_partition'):
        from scripts.neuralese_data.partition_acquired_context_pending import verify_partition, verify_partition_census
        _, pending_count = verify_partition(screen, predecessor_path)
        verify_partition_census(screen, census)
    elif census.get('pending_partition'):
        raise ValueError('pending census lacks its partitioned screen')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('successor output must be a fresh immutable directory')
    if not prepared_source_receipt:
        raise ValueError('a completed source/geometry preparation is required')
    require_output_headroom(output)
    if (screen.get('schema') != 'natlang.acquired-current-protection-screen/1'
            or census.get('schema') != 'natlang.acquired-effective-learning-census/1'
            or census.get('training_admission') is not False):
        raise ValueError('screen or completed effective-loss census is not recognized')
    source_path = checked(previous.get('source'), 'predecessor source')
    map_path = checked(previous.get('family_map'), 'predecessor family map')
    geometry_path = checked(previous.get('geometry'), 'predecessor geometry')
    ce_path = checked(previous.get('ce_policy'), 'predecessor CE policy')
    recipe_path = checked(previous.get('recipe'), 'predecessor recipe')
    checked(census.get('screen'), 'census screen'); checked(census.get('ce_policy'), 'census CE policy')
    census_builder = checked(census.get('builder'), 'census builder')
    if census_builder != (ROOT / 'scripts/neuralese_data/audit_acquired_learning_signal.py').resolve(strict=True):
        raise ValueError('census was not produced by the audited effective-loss program')
    signals_path = checked(census.get('row_signals'), 'census row signals')
    if (Path(census['screen']['path']).resolve() != screen_path or census['ce_policy']['sha256'] != previous['ce_policy']['sha256']):
        raise ValueError('completed census does not bind this screen and actual predecessor CE policy')
    old_policy = read(ce_path)
    objective = policy_objective(old_policy)
    if census.get('objective_sha256') != canonical_json_sha256_hex(objective):
        raise ValueError('census used a different effective loss objective')
    prior_census, mass_custody = predecessor_mass_census(previous, predecessor_path, objective)
    balance = objective.get('family_balance') or {}
    if (objective.get('name') != 'own-outputs/1' or objective.get('mass') != 'token'
            or balance.get('key') != 'logical-family' or balance.get('normalization') != 'declared/1'
            or balance.get('max_scale') != 8 or identity(objective) != previous['objective_transition']['to']):
        raise ValueError('predecessor objective is not the exact declared token-mass continuation')
    protection = ProtectedWorlds.load()
    if (receipt_protection_digest(previous) != protection.digest
            or screen.get('protected_worlds', {}).get('identity_set_sha256') != protection.digest
            or screen.get('protected_worlds', {}).get('matches_expected') is not True):
        raise ValueError('current full protected-world union differs from source custody')
    coverage_path = checked(screen.get('coverage_audit'), 'screen coverage audit')
    coverage = read(coverage_path)
    if (coverage.get('schema') != 'natlang.acquired-session-conversion-coverage-audit/1'
            or coverage.get('exports_manifest_sha256') != screen['coverage_audit'].get('source_manifest_sha256')
            or coverage.get('complete_receipts') != screen['coverage_audit'].get('complete_receipts')
            or screen.get('input_receipts', {}).get('count') != sum(len(r.get('outputs', [])) for r in coverage.get('receipts', []))
            or not isinstance(screen.get('input_receipts', {}).get('sha256'), str)
            or len(screen['input_receipts']['sha256']) != 64):
        raise ValueError('screen does not bind the current conversion inventory')
    accounting = screen.get('accounting') or {}
    collision_count = predecessor_collision_count(screen, predecessor_path, read(map_path)['rows'])
    train_count = sum(entry.get('counts', {}).get('graphs:train', 0) for entry in coverage.get('receipts', []))
    excluded = screen.get('excluded_current_protection')
    if (not isinstance(excluded, list) or accounting.get('train_graphs_before_current_check') != train_count
            or accounting.get('current_protected_exclusions') != len(excluded)
            or train_count - len(excluded) - collision_count - pending_count != screen['output']['rows']
            or accounting.get('eligible_train_graphs') != screen['output']['rows']):
        raise ValueError('screened graph counts do not reconcile with receipt inventory and protection exclusions')
    format_custody = verify_conversion_custody(coverage, screen, screen_program)
    old_map = read(map_path)
    if (old_map.get('schema') != 'natlang.logical-family-map/1' or old_map.get('source_sha256') != previous['source']['sha256']
            or len(old_map.get('rows', [])) != previous['rows']):
        raise ValueError('predecessor family map is not source-bound')
    family_counts = Counter(row['family'] for row in old_map['rows'])
    if {name: item['rows'] for name, item in prior_census['families'].items()} != dict(family_counts):
        raise ValueError('predecessor family census row counts changed')
    old_ids = {row['id'] for row in old_map['rows']}
    if len(old_ids) != previous['rows']:
        raise ValueError('predecessor family map repeats graph identities')
    signals_before = file_signature(signals_path)
    rows, masses, counts, stores = load_signals(signals_path, census=census, screen=screen, protected=protection, old_ids=old_ids)
    diagnosis_proof = verify_diagnosis_proof(rows, diagnosis_verification, diagnosis_program, bind(screen_path), previous['ce_policy'])
    if file_signature(signals_path) != signals_before:
        raise ValueError('completed row census changed while reading')
    mass = sum(masses.values())
    family_plan = scales_for_append(prior_census, balance['scales'], mass, share=share, cap=balance['max_scale'], family=family)
    graph_ref = screen['output']
    graph_path = Path(graph_ref['path']).resolve(strict=True)
    graph_before = file_signature(graph_path)
    if (graph_before[2] != graph_ref['bytes'] or bind(graph_path)['sha256'] != graph_ref['sha256']):
        raise ValueError('screened source file bytes changed')
    tokenizer = read(geometry_path)['identity']['tokenizer']
    roles = read(geometry_path)['identity']['roles']
    if graph_ref['tokenizer_fingerprint'] != tokenizer:
        raise ValueError('screen tokenizer differs from the predecessor')
    if prepared_source_receipt:
        prepared_path = Path(prepared_source_receipt).resolve(strict=True)
        prepared = read(prepared_path)
        if (prepared.get('schema') != PREPARED_SCHEMA or prepared.get('training_admission') is not False
                or prepared.get('predecessor', {}).get('sha256') != bind(predecessor_path)['sha256']
                or prepared.get('screen', {}).get('sha256') != bind(screen_path)['sha256']
                or prepared.get('predecessor_source', {}).get('sha256') != previous['source']['sha256']
                or prepared.get('graph', {}).get('path') != str(graph_path)
                or prepared.get('graph', {}).get('sha256') != graph_ref['sha256']
                or prepared.get('graph', {}).get('bytes') != graph_ref['bytes']
                or prepared.get('tokenizer') != tokenizer or prepared.get('roles') != roles
                or prepared.get('protection_digest') != protection.digest
                or prepared.get('new_rows') != len(rows) or prepared.get('rows') != previous['rows'] + len(rows)):
            raise ValueError('prepared source receipt does not bind the exact predecessor and screen')
        new_source_path = checked(prepared.get('source'), 'prepared source')
        prepared_geometry_path = checked(prepared.get('geometry'), 'prepared geometry')
        prepared_builder = checked_preparation_builder(prepared_path, prepared['builder'])
        parallel_proof = None
        if prepared.get('parallel_index'):
            parallel_proof_path = checked(prepared['parallel_index'], 'parallel source index proof')
            parallel_proof = read(parallel_proof_path)
            checked(parallel_proof.get('parallel_builder'), 'parallel index builder')
            if (parallel_proof.get('schema') != 'natlang.parallel-native-source-index/1'
                    or parallel_proof.get('source', {}).get('sha256') != graph_ref['sha256']
                    or parallel_proof.get('rows') != len(rows) or parallel_proof.get('decision_readouts') != 0):
                raise ValueError('parallel source index has invalid decision census evidence')
        if read(new_source_path) != appended_manifest(source_path, graph_path, graph_ref):
            raise ValueError('prepared manifest does not preserve the exact predecessor plus screened append')
        old_geometry, prepared_geometry = read(geometry_path), read(prepared_geometry_path)
        if (prepared_geometry.get('identity') != old_geometry.get('identity')
                or prepared_geometry.get('kv_slots', [])[:previous['rows']] != old_geometry.get('kv_slots')):
            raise ValueError('prepared geometry changed the exact predecessor geometry prefix')
        if parallel_proof and (prepared_geometry.get('identity', {}).get('parser_policy_sha256') != parallel_proof.get('parser_policy_sha256')
                or canonical_json_sha256_hex(prepared_geometry.get('kv_slots', [])[previous['rows']:]) != parallel_proof.get('ordered_geometry_sha256')):
            raise ValueError('prepared geometry differs from the parallel parser evidence')
        new_source = NativeGraphSource(new_source_path, tokenizer, required_split='train', role_ids=roles)
        actual_new_metadata = [{key: value for key, value in entry.items() if key != 'training_priority'}
                               for entry in new_source.entries[previous['rows']:]]
        if parallel_proof and canonical_json_sha256_hex(actual_new_metadata) != parallel_proof['ordered_metadata_sha256']:
            raise ValueError('prepared source row index differs from the parallel parser evidence')
        output.mkdir(parents=True)
        geometry_values, geometry_ref, geometry_scan = capacity.geometry_for_source(new_source, output, prior=prepared['geometry'], allow_build=False)
    else:
        output.mkdir(parents=True)
        manifest = appended_manifest(source_path, graph_path, graph_ref)
        new_source_path = output / 'train-sources.json'; write_json_durable(new_source_path, manifest)
        new_source, geometry_values, geometry_ref, geometry_scan = capacity.prepare_source(new_source_path, tokenizer, roles, output, prior=previous['geometry'])
    if file_signature(graph_path) != graph_before:
        raise ValueError('screened graph bytes changed during indexing')
    if len(new_source.entries) != previous['rows'] + len(rows):
        raise ValueError('indexed successor row count differs from its census')
    if [(e['id'], e['binding']) for e in new_source.entries[:previous['rows']]] != [(e['id'], e['binding']) for e in old_map['rows']]:
        raise ValueError('predecessor source prefix changed')
    additions = []
    for signal, entry in zip(rows, new_source.entries[previous['rows']:], strict=True):
        if (signal['id'] != entry['id'] or signal['forward_tokens'] != entry['forward_tokens']
                or signal['raw_positive_tokens'] != entry['supervised_tokens']
                or signal['source_groups'] != list(entry['groups']) or entry.get('communication')
                or entry.get('producer_consumer') or entry.get('external_state_consumption')):
            raise ValueError('indexed acquired graph differs from its completed census or native classification')
        additions.append({'id': entry['id'], 'binding': entry['binding'], 'family': family})
    new_map_path = output / 'families.json'
    write_json_durable(new_map_path, {'schema': 'natlang.logical-family-map/1',
        'source_sha256': bind(new_source_path)['sha256'], 'rows': old_map['rows'] + additions})
    policy = copy.deepcopy(old_policy)
    policy['inventory'] = [{k: v for k, v in bind(new_source_path).items() if k != 'bytes'}]
    policy['objective']['family_balance']['map'] = {k: v for k, v in bind(new_map_path).items() if k != 'bytes'}
    policy['objective']['family_balance']['scales'] = family_plan['scales']
    policy['objective_transition'] = {'from': previous['objective_transition']['to']['sha256']}
    if any(item.get('sha256') == graph_ref['sha256'] for item in policy['sources']):
        raise ValueError('acquired source is already declared in the predecessor CE policy')
    policy['sources'].append({'sha256': graph_ref['sha256'], 'encoding': 'native-row-objectives-v1',
        'context_weight': policy['source_contract']['default_raw_context_weight']})
    policy['sources'].sort(key=lambda row: row['sha256'])
    policy['source_contract']['inventory'] = [bind(new_source_path)]
    policy['source_contract']['active_sources'].append({'path': str(graph_path), 'sha256': graph_ref['sha256'], 'bytes': graph_ref['bytes']})
    policy['source_contract']['active_source_count'] = len(policy['source_contract']['active_sources'])
    new_ce_path = output / 'ce-policy.json'; write_json_durable(new_ce_path, policy)
    prepared_policy = prepare_policy(load_policy(bind(new_ce_path)), new_source)
    unscaled_policy = copy.deepcopy(policy); unscaled_policy['objective']['family_balance'] = None
    unscaled_policy_path = output / 'unscaled-ce-policy.json'; write_json_durable(unscaled_policy_path, unscaled_policy)
    all_family_counts = Counter(row['family'] for row in old_map['rows'] + additions)
    mass_census_path = output / 'objective-census.json'
    write_json_durable(mass_census_path, {'schema': 'natlang.own-output-mass-census/1',
        'manifest': bind(new_source_path), 'policy': bind(unscaled_policy_path), 'rows': len(new_source.entries),
        'objective_without_family_scales': True,
        'families': {name: {'rows': count, 'unscaled_mass': family_plan['unscaled_mass'][name]}
                     for name, count in sorted(all_family_counts.items())},
        'unscaled_mass_total': family_plan['total_unscaled_mass'],
        'predecessor_mass_custody': mass_custody, 'delta_census': bind(census_path),
        'scope': 'Exact sum of measured predecessor and appended unscaled masses; no repeated full-corpus scan.'})
    mass_receipt_path = output / 'objective-census-receipt.json'
    write_json_durable(mass_receipt_path, {'schema': 'natlang.measured-mass-append-receipt/1',
        'census': bind(mass_census_path), 'map': bind(new_map_path), 'builder': bind(Path(__file__))})
    descriptor = output / 'sft-capacity-descriptor.json'
    geometry_summary = {'max_forward_tokens': max(entry['forward_tokens'] for entry in new_source.entries),
        'max_kv_slots': max(geometry_values), 'forward_tokens_over_262144':
            sum(entry['forward_tokens'] > 262144 for entry in new_source.entries),
        'new_max_forward_tokens': max(entry['forward_tokens'] for entry in new_source.entries[previous['rows']:]),
        'new_forward_tokens_over_262144': sum(entry['forward_tokens'] > 262144
            for entry in new_source.entries[previous['rows']:]),
        'scope': 'Untruncated indexed graph geometry; not serving-context or training-position qualification.'}
    write_json_durable(descriptor, sft_capacity_descriptor(bind(new_source_path), geometry_ref))
    recipe = read(recipe_path)
    recipe['input_bindings']['v4_previous_train_graphs'] = {k: v for k, v in bind(source_path).items() if k != 'bytes'}
    recipe['input_bindings']['v4_train_graphs'] = {k: v for k, v in bind(new_source_path).items() if k != 'bytes'}
    recipe['input_bindings']['v4_ce_policy'] = {k: v for k, v in bind(new_ce_path).items() if k != 'bytes'}
    operational_stage = next(item for item in recipe['stages'] if item['id'] == 'native_joint_sft_v4')
    operational_stage['parameters']['sft_packed_geometry'] = geometry_ref
    operational_stage['parameters']['memory_qualification'] = {k: v for k, v in bind(descriptor).items() if k != 'bytes'}
    new_recipe_path = output / 'recipe-continuation.json'; write_json_durable(new_recipe_path, recipe)
    old_recipe, new_recipe = load_recipe(recipe_path), load_recipe(new_recipe_path)
    stage = lambda value: next(item for item in value['stages'] if item['id'] == 'native_joint_sft_v4')
    old_roles = resolve_stage_inputs(old_recipe, stage(old_recipe), {})
    new_roles = resolve_stage_inputs(new_recipe, stage(new_recipe), {})
    if not _data_continuation({k: v for k, v in old_roles.items() if k != 'ce_supervision_policy'},
                              {k: v for k, v in new_roles.items() if k != 'ce_supervision_policy'}):
        raise ValueError('candidate does not satisfy the authenticated recipe continuation')
    for role in ('native_graphs', 'foundation_native_graphs'):
        if (Path(old_roles[role]['path']).resolve() != source_path or Path(new_roles[role]['path']).resolve() != new_source_path):
            raise ValueError(f'shared {role} role did not move with the other source role')
    if old_roles.get('held_native_graphs') != new_roles.get('held_native_graphs'):
        raise ValueError('candidate changed the held source binding')
    old_index = SimpleNamespace(file_sha256=previous['source']['sha256'], entries=new_source.entries[:previous['rows']])
    extension = extension_policy(old_roles, new_roles, prior_train=old_index, candidate_train=new_source)
    _, transition = continuation_identity({'supervision_objective': previous['objective_transition']['to']}, extension)
    if file_signature(signals_path) != signals_before or file_signature(graph_path) != graph_before:
        raise ValueError('source or census changed while staging the continuation')
    result = {'schema': SCHEMA, 'status': 'source-ce-recipe-geometry-qualified-not-published', 'training_admission': False,
        'predecessor': bind(predecessor_path), 'predecessor_source': bind(source_path), 'source_screen': bind(screen_path),
        'coverage_audit': bind(coverage_path), 'effective_loss_census': bind(census_path), 'row_signals': bind(signals_path),
        'source': bind(new_source_path), 'appended_graphs': {'path': str(graph_path), 'sha256': graph_ref['sha256'], 'bytes': graph_ref['bytes']},
        'family_map': bind(new_map_path), 'ce_policy': bind(new_ce_path), 'recipe': bind(new_recipe_path),
        'objective_census': bind(mass_census_path), 'objective_census_receipt': bind(mass_receipt_path),
        'geometry': geometry_ref, 'capacity_descriptor': bind(descriptor), 'observed_geometry': geometry_summary, 'rows': len(new_source.entries),
        'predecessor_rows': previous['rows'], 'new_rows': len(rows), 'new_family': family,
        'counts': counts, 'category_mass': masses, 'source_stores': stores, 'family_balance': family_plan,
        'source_format_custody': format_custody,
        'source_diagnosis_target_proof': diagnosis_proof,
        'family_census': prepared_policy['_family_census'], 'objective_transition': transition,
        'geometry_scan': geometry_scan, 'current_protection_digest': protection.digest,
        'prepared_source': bind(prepared_source_receipt) if prepared_source_receipt else None,
        'prepared_source_builder': prepared_builder if prepared_source_receipt else None,
        'roles': {key: new_roles.get(key) for key in ('native_graphs', 'foundation_native_graphs', 'held_native_graphs')},
        'builder': sealed_builder(output), 'limitations': ['No local oracle replay or behavioral qualification was performed.',
            'Source-curated labels, including diagnosis labels, remain source evidence; no learner exposure is claimed.',
            'Recipe parameters are inherited from the source predecessor; reconcile them with the actual live learner and runtime before an operational continuation.']}
    receipt = output / 'receipt.json'; write_json_durable(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-receipt', type=Path, required=True)
    parser.add_argument('--screen-receipt', type=Path, required=True)
    parser.add_argument('--census-receipt', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-source-only', action='store_true')
    parser.add_argument('--parallel-workers', type=int)
    parser.add_argument('--prepared-source-receipt', type=Path)
    parser.add_argument('--diagnosis-verification', type=Path)
    parser.add_argument('--diagnosis-program', type=Path)
    parser.add_argument('--screen-program', type=Path)
    parser.add_argument('--new-family-share', type=float, default=.07)
    parser.add_argument('--family', default=FAMILY)
    args = parser.parse_args()
    if args.prepare_source_only:
        if args.census_receipt or args.prepared_source_receipt or args.diagnosis_verification or args.diagnosis_program or args.screen_program:
            parser.error('source-only preparation takes no census or earlier preparation')
        result = prepare_source_stage(args.predecessor_receipt, args.screen_receipt, args.output_dir,
                                      parallel_workers=2 if args.parallel_workers is None else args.parallel_workers)
    else:
        if not args.census_receipt or not args.screen_program or not args.prepared_source_receipt:
            parser.error('completed --census-receipt, --prepared-source-receipt and --screen-program are required for the successor')
        if args.parallel_workers is not None:
            parser.error('--parallel-workers is only for source preparation')
        result = build(args.predecessor_receipt, args.screen_receipt, args.census_receipt,
                       args.output_dir, share=args.new_family_share, prepared_source_receipt=args.prepared_source_receipt,
                       diagnosis_verification=args.diagnosis_verification, diagnosis_program=args.diagnosis_program,
                       family=args.family, screen_program=args.screen_program)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
