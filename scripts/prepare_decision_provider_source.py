#!/usr/bin/env python3
"""Prepare a bounded, provenance-pinned subset of existing typed choice cases.

This script only prepares source inputs. It does not call a provider, launch a
worker, or grant training admission. Original case bytes/row hashes remain
pinned; a declared shared family criterion may be added as a separate derived
model-visible field. All filtering is represented in omissions.jsonl.
"""
import argparse
import collections
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402
from natlang_neuralese.common.hashing import sha256_hex as sha256_bytes  # noqa: E402
from decision_task_contracts import CONTRACT_PATH, criteria_for, load_contracts  # noqa: E402


def parse_families(values):
    families = []
    for value in values:
        families.extend(part.strip() for part in value.split(',') if part.strip())
    result = list(dict.fromkeys(families))
    if not result:
        raise ValueError('at least one --families entry is required')
    return result


def source_ref_summary(case):
    """Keep source row identity in omissions without copying its full state."""
    return {
        'id': case.get('id'),
        'group': case.get('group'),
        'source': case.get('source'),
        'family': case.get('family'),
        'split': case.get('role'),
        'source_refs': case.get('source_refs')
    }


def load_quality_holds(paths):
    """Merge canonical and caller-supplied holds without allowing overrides."""
    holds, receipts, canonical_policy_doc = {}, [], None
    for raw_path, is_canonical in paths:
        if not raw_path:
            continue
        hold_path = Path(raw_path).resolve()
        raw = hold_path.read_bytes()
        doc = json.loads(raw)
        if not isinstance(doc, dict) or not isinstance(doc.get('excluded_items'), list):
            raise ValueError(f'{hold_path}: source-quality holds must contain an excluded_items array')
        if is_canonical and doc.get('schema') != 'natlang.decision-source-quality-holds/1':
            raise ValueError(f'{hold_path}: invalid canonical source-quality policy schema')
        if is_canonical:
            rules = doc.get('state_truncation_rules')
            if not isinstance(rules, list):
                raise ValueError(f'{hold_path}: canonical policy must declare state_truncation_rules')
            for rule in rules:
                if (not isinstance(rule, dict) or not isinstance(rule.get('rule_id'), str) or
                        not isinstance(rule.get('state_codepoints'), int) or rule['state_codepoints'] <= 0 or
                        not isinstance(rule.get('suffix'), str) or not rule['suffix']):
                    raise ValueError(f'{hold_path}: malformed state truncation rule')
            canonical_policy_doc = doc
        receipt = {'path': str(hold_path), 'sha256': sha256_bytes(raw), 'schema': doc.get('schema'),
                   'canonical_policy': is_canonical}
        receipts.append(receipt)
        for item in doc['excluded_items']:
            if not isinstance(item, dict) or not isinstance(item.get('item_id'), str) or not isinstance(item.get('reason'), str):
                raise ValueError(f'{hold_path}: each source-quality exclusion needs item_id and reason')
            item_group = item.get('original_source_group', item.get('group'))
            item_split = item.get('split')
            if is_canonical and (not item_group or not item_split):
                raise ValueError(f"{hold_path}: canonical hold needs source group and split for {item['item_id']}")
            if item_group is not None and not isinstance(item_group, str):
                raise ValueError(f"{hold_path}: invalid group for {item['item_id']}")
            if item_split is not None and not isinstance(item_split, str):
                raise ValueError(f"{hold_path}: invalid split for {item['item_id']}")
            entry = holds.setdefault(item['item_id'], {'group': item_group, 'split': item_split, 'holds': []})
            if entry['group'] is not None and item_group is not None and entry['group'] != item_group:
                raise ValueError(f"{item['item_id']}: quality hold receipts disagree on source group")
            if entry['split'] is not None and item_split is not None and entry['split'] != item_split:
                raise ValueError(f"{item['item_id']}: quality hold receipts disagree on split")
            entry['group'] = entry['group'] or item_group
            entry['split'] = entry['split'] or item_split
            held = {'reason': item['reason'], 'receipt_item_id': item['item_id'],
                    'receipt_group': item_group, 'receipt_split': item_split,
                    'receipt_sha256': receipt['sha256'], 'receipt_path': receipt['path'],
                    'canonical_policy': is_canonical, 'evidence': item.get('evidence', {}),
                    'official_row_sha256': item.get('official_row_sha256')}
            if held not in entry['holds']:
                entry['holds'].append(held)
    return holds, receipts, canonical_policy_doc


def omission(case, source_path, source_sha, line_number, row_sha, reasons):
    return {
        'source_file': str(source_path),
        'source_file_sha256': source_sha,
        'source_line_number': line_number,
        'source_row_sha256': row_sha,
        'source_row': source_ref_summary(case),
        'reasons': reasons
    }


def read_prior_cases(prior_root):
    root = Path(prior_root).resolve()
    files = sorted(root.glob('provider-*/cases.jsonl'))
    program_ids, program_groups = set(), set()
    source_ids, source_groups = set(), set()
    manifests = []
    for path in files:
        file_program_ids, file_program_groups = set(), set()
        file_source_ids, file_source_groups = set(), set()
        raw_sha = sha256_file(path)
        with path.open('rb') as stream:
            for line_no, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                case = json.loads(raw_line)
                if isinstance(case.get('id'), str):
                    file_program_ids.add(case['id'])
                    program_ids.add(case['id'])
                if isinstance(case.get('group'), str):
                    file_program_groups.add(case['group'])
                    program_groups.add(case['group'])
                for field, target, file_target in (
                        ('source_ids', source_ids, file_source_ids),
                        ('source_groups', source_groups, file_source_groups)):
                    values = case.get(field)
                    if values is None:
                        continue
                    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                        raise ValueError(f'{path}:{line_no}: {field} must be an array of nonempty strings')
                    target.update(values)
                    file_target.update(values)
        manifests.append({
            'path': str(path), 'sha256': raw_sha,
            # Retain the old keys as explicit direct program-level counts.
            'unique_ids': len(file_program_ids), 'unique_groups': len(file_program_groups),
            'direct_program_ids': len(file_program_ids), 'direct_program_groups': len(file_program_groups),
            'nested_source_ids': len(file_source_ids), 'nested_source_groups': len(file_source_groups)
        })
    return root, program_ids, program_groups, source_ids, source_groups, manifests


def source_manifest_pin(source_path, source_sha, source_rows):
    """Read the adjacent source manifest when available for exact cross-packet lineage."""
    candidates = [source_path.parent / 'source-manifest.json', source_path.parent / 'decision-data.manifest.json']
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return None, None, None
    raw = manifest_path.read_bytes()
    doc = json.loads(raw)
    declared_count = doc.get('cases', {}).get('rows') if isinstance(doc.get('cases'), dict) else doc.get('cases')
    declared_hash = (doc.get('cases', {}).get('sha256') if isinstance(doc.get('cases'), dict) else None)
    declared_hash = declared_hash or doc.get('selected_sha256') or doc.get('cases_sha256')
    if declared_hash is None and isinstance(doc.get('sha256'), dict):
        declared_hash = doc['sha256'].get(source_path.name)
    declared_count = declared_count or doc.get('selected_count') or doc.get('selection', {}).get('selected_count')
    if declared_hash != source_sha or declared_count != source_rows:
        raise ValueError(f'{manifest_path}: source manifest does not pin the supplied source bytes/count')
    return manifest_path.resolve(), sha256_bytes(raw), doc


def load_reserved_annotation_cases(paths, source_path, source_sha, source_manifest_sha):
    """Validate annotation-folder reservations against exact source rows, then reserve IDs/groups."""
    if not paths:
        return set(), set(), {}, {}, []
    row_pins = {}
    id_pins = {}
    with source_path.open('rb') as stream:
        source_index = 0
        for raw_line in stream:
            if not raw_line.strip():
                continue
            row_sha = sha256_bytes(raw_line.rstrip(b'\r\n'))
            case = json.loads(raw_line)
            entry = (case.get('id'), case.get('group'), case.get('role'), row_sha)
            row_pins[source_index] = entry
            if isinstance(case.get('id'), str):
                if case['id'] in id_pins:
                    raise ValueError(f'{source_path}: duplicate source ID {case["id"]}')
                id_pins[case['id']] = (source_index, entry)
            source_index += 1
    reserved_ids, reserved_groups, reserved_id_refs, reserved_group_refs, receipts = set(), set(), {}, {}, []
    reserved_program_ids = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        file_sha = sha256_file(path)
        programs = 0
        file_ids, file_groups = set(), set()
        with path.open('rb') as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                program = json.loads(raw_line)
                programs += 1
                program_id = program.get('id')
                if not isinstance(program_id, str) or not program_id or program_id in reserved_program_ids:
                    raise ValueError(f'{path}:{line_number}: missing or duplicate reserved program ID')
                reserved_program_ids.add(program_id)
                external = program.get('external_source')
                if not isinstance(external, dict) or (external.get('snapshot_sha256') != source_sha or
                        external.get('source_manifest_sha256') != source_manifest_sha):
                    raise ValueError(f'{path}:{line_number}: reservation source snapshot mismatch')
                ids = program.get('source_ids')
                groups = program.get('source_groups')
                rows = external.get('source_rows')
                if not isinstance(ids, list) or not isinstance(groups, list) or not isinstance(rows, list) or len(ids) != len(rows):
                    raise ValueError(f'{path}:{line_number}: malformed reservation source lineage')
                actual_groups = []
                for source_id, declared in zip(ids, rows):
                    if not isinstance(source_id, str) or not isinstance(declared, dict) or declared.get('id') != source_id:
                        raise ValueError(f'{path}:{line_number}: reservation ID/source-row mismatch')
                    source_index = declared.get('row_index')
                    if not isinstance(source_index, int) or source_index not in row_pins:
                        raise ValueError(f'{path}:{line_number}: reservation row index is invalid')
                    actual = row_pins[source_index]
                    if (actual[0] != source_id or actual[1] != declared.get('group') or
                            actual[2] != declared.get('split') or actual[3] != declared.get('row_sha256') or
                            actual[2] != 'train'):
                        raise ValueError(f'{path}:{line_number}: reservation does not match exact source row {source_id}')
                    if source_id in reserved_ids or actual[1] in reserved_groups:
                        raise ValueError(f'{path}:{line_number}: duplicate reserved source ID/group')
                    reserved_ids.add(source_id)
                    reserved_groups.add(actual[1])
                    reserved_id_refs[source_id] = {'path': str(path), 'program_id': program_id}
                    reserved_group_refs[actual[1]] = {'path': str(path), 'program_id': program_id}
                    file_ids.add(source_id)
                    file_groups.add(actual[1])
                    actual_groups.append(actual[1])
                if len(set(actual_groups)) != len(groups) or set(actual_groups) != set(groups):
                    raise ValueError(f'{path}:{line_number}: reservation source_groups mismatch')
        receipts.append({'path': str(path), 'sha256': file_sha, 'programs': programs,
                         'source_ids': len(file_ids), 'source_groups': len(file_groups),
                         'identity_binding': 'each source_id is bound to an exact source line index, row SHA-256, group, and train split'})
    return reserved_ids, reserved_groups, reserved_id_refs, reserved_group_refs, receipts


def build(args):
    families = parse_families(args.families)
    source_path = Path(args.source).resolve()
    prior_root, prior_program_ids, prior_program_groups, prior_source_ids, prior_source_groups, prior_files = read_prior_cases(args.prior_root)
    used_ids = prior_program_ids | prior_source_ids
    used_groups = prior_program_groups | prior_source_groups
    source_bytes_sha = sha256_file(source_path)
    with source_path.open('rb') as source_stream:
        source_rows = sum(1 for line in source_stream if line.strip())
    source_manifest_path, source_manifest_sha, _ = source_manifest_pin(source_path, source_bytes_sha, source_rows)
    if args.exclude_cases and not source_manifest_sha:
        raise ValueError('annotation reservations require an adjacent exact source manifest')
    reserved_ids, reserved_groups, reserved_id_refs, reserved_group_refs, reservation_receipts = load_reserved_annotation_cases(
        args.exclude_cases, source_path, source_bytes_sha, source_manifest_sha)
    canonical_policy_path = Path(__file__).resolve().parents[1] / 'training' / 'decision_source_quality_holds.json'
    quality_holds, quality_receipts, canonical_policy = load_quality_holds([
        (canonical_policy_path, True), (args.source_quality_holds, False)
    ])
    truncation_rules = canonical_policy['state_truncation_rules']
    preparation_code_sha = sha256_file(Path(__file__).resolve())
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(f'output directory must be fresh: {destination}')

    family_set = set(families)
    pools = {family: [] for family in families}
    omissions = []
    source_rows = 0
    source_scope_rows = 0
    source_ids_seen = {}
    seen_quality_hold_ids = set()
    source_row_keys = {}

    with source_path.open('rb') as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            source_rows += 1
            row_sha = sha256_bytes(raw_line.rstrip(b'\r\n'))
            try:
                case = json.loads(raw_line)
            except Exception as exc:
                raise ValueError(f'{source_path}:{line_number}: invalid JSON: {exc}') from exc
            if not isinstance(case, dict):
                continue
            case_id = case.get('id')
            if isinstance(case_id, str):
                if case_id in source_ids_seen:
                    raise ValueError(f'{source_path}:{line_number}: duplicate source case ID {case_id}')
                source_ids_seen[case_id] = line_number
            family = case.get('family')
            if family not in family_set or case.get('role') != 'train' or case.get('kind') != 'choice':
                continue
            source_scope_rows += 1
            source_row_keys[case_id] = (case.get('group'), case.get('role'), row_sha)
            reasons = []
            if case_id in quality_holds:
                hold = quality_holds[case_id]
                expected_group = hold['group']
                expected_split = hold['split']
                if expected_group is not None and expected_group != case.get('group'):
                    raise ValueError(f'{case_id}: quality hold group does not match source row')
                if expected_split is not None and expected_split != case.get('role'):
                    raise ValueError(f'{case_id}: quality hold split does not match source row')
                reasons.extend({
                    'code': 'explicit_source_quality_hold', 'detail': item['reason'],
                    'receipt_item_id': item['receipt_item_id'], 'receipt_group': expected_group,
                    'receipt_split': expected_split, 'receipt_path': item['receipt_path'],
                    'receipt_sha256': item['receipt_sha256'], 'canonical_policy': item['canonical_policy'],
                    'evidence': item['evidence'], 'upstream_row_sha256': item['official_row_sha256'],
                    'policy_sha256': quality_receipts[0]['sha256'],
                    'preparation_code_sha256': preparation_code_sha
                } for item in hold['holds'])
                seen_quality_hold_ids.add(case_id)
            state = case.get('state')
            options = case.get('options')
            if isinstance(state, str):
                for rule in truncation_rules:
                    if len(state) == rule['state_codepoints'] and state.endswith(rule['suffix']):
                        reasons.append({'code': rule['rule_id'], 'state_chars': len(state),
                                        'state_codepoints': len(state), 'rule_id': rule['rule_id']})
            if not isinstance(state, str) or not state.strip():
                reasons.append({'code': 'source_quality_invalid_or_empty_state'})
            elif len(state) > args.max_state_chars:
                reasons.append({'code': 'state_exceeds_max_chars', 'state_chars': len(state),
                                'max_state_chars': args.max_state_chars})
            if not isinstance(options, list) or any(not isinstance(option, str) for option in options):
                reasons.append({'code': 'source_quality_invalid_options'})
            elif not 2 <= len(options) <= args.max_options:
                reasons.append({'code': 'option_count_out_of_range', 'option_count': len(options),
                                'min_options': 2, 'max_options': args.max_options})
            elif len(set(options)) != len(options):
                reasons.append({'code': 'source_quality_duplicate_options'})
            if not isinstance(case.get('answer'), str) or (isinstance(options, list) and case.get('answer') not in options):
                reasons.append({'code': 'source_quality_answer_not_declared_option'})
            for key in ('id', 'group', 'source', 'question'):
                if not isinstance(case.get(key), str) or not case[key]:
                    reasons.append({'code': f'source_quality_missing_{key}'})
            if case_id in prior_program_ids:
                reasons.append({'code': 'prior_provider_program_id_overlap'})
            if case_id in prior_source_ids:
                reasons.append({'code': 'prior_provider_source_id_overlap'})
            if isinstance(case.get('group'), str) and case['group'] in prior_program_groups:
                reasons.append({'code': 'prior_provider_program_group_overlap'})
            if isinstance(case.get('group'), str) and case['group'] in prior_source_groups:
                reasons.append({'code': 'prior_provider_source_group_overlap'})
            if case_id in reserved_ids:
                reasons.append({'code': 'reserved_annotation_source_id',
                                'reservation': reserved_id_refs[case_id]})
            if isinstance(case.get('group'), str) and case['group'] in reserved_groups:
                reasons.append({'code': 'reserved_annotation_source_group',
                                'reservation': reserved_group_refs[case['group']]})
            if reasons:
                omissions.append(omission(case, source_path, source_bytes_sha, line_number, row_sha, reasons))
                continue
            pools[family].append({
                'case': case, 'source_line_number': line_number,
                'source_row_sha256': row_sha
            })

    optional_hold_ids = {item_id for item_id in quality_holds
                         if any(not hold['canonical_policy'] for hold in quality_holds[item_id]['holds'])}
    stale_holds = sorted(optional_hold_ids - seen_quality_hold_ids)
    if stale_holds:
        raise ValueError('source-quality hold receipt contains item IDs not found in the requested train choice scope: '
                         + ', '.join(stale_holds))

    selected = []
    selected_ids, selected_groups = set(used_ids), set(used_groups)
    while len(selected) < args.count:
        progressed = False
        for family in families:
            if len(selected) >= args.count:
                break
            candidates = pools[family]
            while candidates:
                candidate = candidates.pop(0)
                case = candidate['case']
                reasons = []
                if case['id'] in selected_ids:
                    reasons.append({'code': 'selected_provider_case_id_collision'})
                if case['group'] in selected_groups:
                    reasons.append({'code': 'selected_provider_source_group_collision'})
                if reasons:
                    omissions.append(omission(case, source_path, source_bytes_sha,
                                              candidate['source_line_number'], candidate['source_row_sha256'], reasons))
                    continue
                selected.append(candidate)
                selected_ids.add(case['id'])
                selected_groups.add(case['group'])
                progressed = True
                break
        if not progressed:
            break

    destination.mkdir(parents=True, exist_ok=False)
    case_path = destination / 'cases.jsonl'
    omission_path = destination / 'omissions.jsonl'
    task_contract_doc, task_contract_sha = load_contracts()
    prepared_cases = []
    derived_criteria_rows = []
    for candidate in selected:
        source_case = candidate['case']
        prepared_case = dict(source_case)
        if source_case.get('kind') == 'choice':
            labels = source_case.get('options')
            criteria, provenance = criteria_for(family=source_case.get('family'),
                source=source_case.get('source'), kind=source_case.get('kind'), labels=labels,
                explicit=source_case.get('criteria'), where=str(source_case.get('id')))
            if provenance and (provenance['family_contract_applied'] or 'criteria' in source_case):
                prepared_case['criteria'] = criteria
                derived_criteria_rows.append({
                    'id': source_case.get('id'), 'source_row_sha256': candidate['source_row_sha256'],
                    'criteria_sha256': sha256_bytes(json.dumps(criteria, sort_keys=True,
                        ensure_ascii=False, separators=(',', ':')).encode('utf-8')),
                    **provenance,
                })
        prepared_cases.append(prepared_case)
    case_data = ''.join(json.dumps(case, sort_keys=True, ensure_ascii=False) + '\n'
                        for case in prepared_cases).encode('utf-8')
    omission_data = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n'
                            for row in omissions).encode('utf-8')
    case_path.write_bytes(case_data)
    omission_path.write_bytes(omission_data)

    by_family = collections.Counter(candidate['case']['family'] for candidate in selected)
    omission_codes = collections.Counter(reason['code'] for row in omissions for reason in row['reasons'])
    manifest = {
        'schema': 'natlang.provider-decision-source-preparation/1',
        'status': 'prepared_only_not_launched',
        'preparation_only': True,
        'training_admission': False,
        'independent_new_worlds': 0,
        'source': {
            'path': str(source_path), 'sha256': source_bytes_sha, 'row_count': source_rows,
            'manifest_path': str(source_manifest_path) if source_manifest_path else None,
            'manifest_sha256': source_manifest_sha,
            'scope_row_count': source_scope_rows,
            'row_hash_contract': 'SHA-256 of exact UTF-8 JSONL row bytes excluding the line terminator'
        },
        'model_visible_task_criteria': {
            'contract_path': str(CONTRACT_PATH.resolve()),
            'contract_sha256': task_contract_sha,
            'contract_schema': task_contract_doc['schema'],
            'loader_path': str(Path(__file__).with_name('decision_task_contracts.py').resolve()),
            'loader_sha256': sha256_file(Path(__file__).with_name('decision_task_contracts.py')),
            'derived_rows': derived_criteria_rows,
            'selected_original_rows': [{'id': item['case'].get('id'),
                'source_line_number': item['source_line_number'], 'source_row_sha256': item['source_row_sha256'],
                'group': item['case'].get('group')} for item in selected],
            'source_integrity': 'Original source bytes and row SHA-256 values are preserved in source pins/selected_rows; criteria are an explicit derived model-visible field in prepared cases.jsonl.',
        },
        'source_quality_holds': {
            'policy_path': str(canonical_policy_path.resolve()),
            'policy_sha256': quality_receipts[0]['sha256'],
            'state_truncation_rules': truncation_rules,
            'preparation_code_path': str(Path(__file__).resolve()),
            'preparation_code_sha256': preparation_code_sha,
            'merge_semantics': 'Canonical holds are always applied; optional receipts add holds. Duplicate IDs merge distinct reasons/evidence after group/split agreement checks.',
            'receipts': quality_receipts
        },
        'selection': {
            'families_round_robin_order': families,
            'requested_count': args.count,
            'selected_count': len(selected),
            'shortfall': args.count - len(selected),
            'max_state_chars': args.max_state_chars,
            'max_options': args.max_options,
            'required_role': 'train', 'required_kind': 'choice',
            'disjointness': 'case IDs and source groups are checked against every prior-root provider-*/cases.jsonl and against the selected rows',
            'preservation': 'selected source answers, IDs, groups, and original row hashes are preserved; only explicitly derived shared criteria may be added to prepared model input rows',
            'order': 'source file order within each family, selected in repeated round-robin passes'
        },
        'prior_provider_cases': {
            'root': str(prior_root), 'files': prior_files,
            'unique_case_ids': len(prior_program_ids), 'unique_source_groups': len(prior_program_groups),
            'unique_direct_program_ids': len(prior_program_ids),
            'unique_direct_program_groups': len(prior_program_groups),
            'unique_nested_source_ids': len(prior_source_ids),
            'unique_nested_source_groups': len(prior_source_groups),
            'unique_reserved_id_union': len(used_ids), 'unique_reserved_group_union': len(used_groups),
            'lineage_policy': 'Direct program IDs/groups and nested source_ids/source_groups are stored and counted separately, then the union is used for overlap prevention.'
        },
        'annotation_reservations': {
            'files': reservation_receipts,
            'reserved_source_ids': len(reserved_ids),
            'reserved_source_groups': len(reserved_groups),
            'policy': 'Each reservation is validated against the exact source manifest and every exact source row index/hash/group/train split; IDs and groups are excluded from selection.'
        },
        'by_family': dict(sorted(by_family.items())),
        'omissions': {
            'path': omission_path.name, 'sha256': sha256_bytes(omission_data), 'rows': len(omissions),
            'reason_counts': dict(sorted(omission_codes.items()))
        },
        'cases': {'path': case_path.name, 'sha256': sha256_bytes(case_data), 'rows': len(selected)},
        'provenance_limit': 'source row IDs/groups and this legacy decision-cases.jsonl file are pinned; upstream dataset revisions are not asserted unless present in each preserved case source_refs',
        'gold_handling': 'source answer fields are preserved in the selected source file for offline scoring; callers must omit them from provider requests',
        'scheduling': 'No provider or worker was invoked; root performs all launch review.'
    }
    manifest_path = destination / 'source-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Exact source decision-cases.jsonl')
    parser.add_argument('--prior-root', required=True, help='Root whose provider-*/cases.jsonl files reserve IDs/groups')
    parser.add_argument('--out', required=True, help='Fresh output directory; existing directories are refused')
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--max-state-chars', type=int, default=3000)
    parser.add_argument('--max-options', type=int, default=20)
    parser.add_argument('--families', action='append', required=True,
                        help='Family name or comma-separated family names; may be repeated')
    parser.add_argument('--source-quality-holds',
                        help='Optional additional JSON hold receipt; canonical training policy is always applied')
    parser.add_argument('--exclude-cases', action='append', default=[],
                        help='Prior annotation program JSONL; validates and reserves its exact source IDs/groups; may be repeated')
    args = parser.parse_args()
    if args.count <= 0 or args.max_state_chars <= 0 or args.max_options < 2:
        parser.error('--count and --max-state-chars must be positive; --max-options must be at least 2')
    try:
        manifest = build(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'prepare_decision_provider_source: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({
        'out': str(Path(args.out).resolve()),
        'selected': manifest['cases']['rows'],
        'requested': manifest['selection']['requested_count'],
        'shortfall': manifest['selection']['shortfall'],
        'cases_sha256': manifest['cases']['sha256'],
        'omissions_sha256': manifest['omissions']['sha256'],
        'by_family': manifest['by_family'],
        'omission_reason_counts': manifest['omissions']['reason_counts'],
        'training_admission': False,
        'status': manifest['status']
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
