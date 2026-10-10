#!/usr/bin/env python3
"""Prepare a bounded, provenance-pinned subset of existing typed choice cases.

This script only prepares source inputs. It does not call a provider, launch a
worker, or grant training admission. Source case objects are copied unchanged;
all filtering is represented in omissions.jsonl.
"""
import argparse
import collections
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402
from natlang_neuralese.common.hashing import sha256_hex as sha256_bytes  # noqa: E402


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
    used_ids, used_groups = set(), set()
    manifests = []
    for path in files:
        file_ids, file_groups = set(), set()
        raw_sha = sha256_file(path)
        with path.open('rb') as stream:
            for line_no, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                case = json.loads(raw_line)
                if isinstance(case.get('id'), str):
                    file_ids.add(case['id'])
                    used_ids.add(case['id'])
                if isinstance(case.get('group'), str):
                    file_groups.add(case['group'])
                    used_groups.add(case['group'])
        manifests.append({
            'path': str(path), 'sha256': raw_sha,
            'unique_ids': len(file_ids), 'unique_groups': len(file_groups)
        })
    return root, used_ids, used_groups, manifests


def build(args):
    families = parse_families(args.families)
    source_path = Path(args.source).resolve()
    prior_root, used_ids, used_groups, prior_files = read_prior_cases(args.prior_root)
    source_bytes_sha = sha256_file(source_path)
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
            if case_id in used_ids:
                reasons.append({'code': 'prior_provider_case_id_overlap'})
            if isinstance(case.get('group'), str) and case['group'] in used_groups:
                reasons.append({'code': 'prior_provider_source_group_overlap'})
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
    case_data = ''.join(json.dumps(candidate['case'], sort_keys=True, ensure_ascii=False) + '\n'
                        for candidate in selected).encode('utf-8')
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
            'scope_row_count': source_scope_rows,
            'row_hash_contract': 'SHA-256 of exact UTF-8 JSONL row bytes excluding the line terminator'
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
            'preservation': 'selected source case objects are serialized without field edits; no gold is added or removed',
            'order': 'source file order within each family, selected in repeated round-robin passes'
        },
        'prior_provider_cases': {
            'root': str(prior_root), 'files': prior_files,
            'unique_case_ids': len(used_ids), 'unique_source_groups': len(used_groups)
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
