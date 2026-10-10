#!/usr/bin/env python3
"""Prepare a bounded, provenance-pinned subset of existing typed choice cases.

This script only prepares source inputs. It does not call a provider, launch a
worker, or grant training admission. Source case objects are copied unchanged;
all filtering is represented in omissions.jsonl.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


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


def load_quality_holds(path):
    if not path:
        return {}, None
    hold_path = Path(path).resolve()
    raw = hold_path.read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or not isinstance(doc.get('excluded_items'), list):
        raise ValueError('source-quality hold receipt must contain an excluded_items array')
    holds = {}
    for item in doc['excluded_items']:
        if not isinstance(item, dict) or not isinstance(item.get('item_id'), str) or not isinstance(item.get('reason'), str):
            raise ValueError('each source-quality exclusion needs item_id and reason')
        if item['item_id'] in holds:
            raise ValueError(f"duplicate source-quality hold for {item['item_id']}")
        holds[item['item_id']] = item
    return holds, {'path': str(hold_path), 'sha256': sha256_bytes(raw), 'schema': doc.get('schema')}


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
    quality_holds, quality_receipt = load_quality_holds(args.source_quality_holds)
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
                expected_group = hold.get('original_source_group')
                expected_split = hold.get('split')
                if expected_group is not None and expected_group != case.get('group'):
                    raise ValueError(f'{case_id}: quality hold group does not match source row')
                if expected_split is not None and expected_split != case.get('role'):
                    raise ValueError(f'{case_id}: quality hold split does not match source row')
                reasons.append({
                    'code': 'explicit_source_quality_hold', 'detail': hold['reason'],
                    'receipt_item_id': hold['item_id'], 'receipt_group': expected_group,
                    'receipt_split': expected_split,
                    'upstream_row_sha256': hold.get('official_row_sha256')
                })
                seen_quality_hold_ids.add(case_id)
            state = case.get('state')
            options = case.get('options')
            if isinstance(state, str) and len(state) == 1504 and state.endswith(' ...'):
                reasons.append({'code': 'legacy_1500_char_truncation_signature', 'state_chars': len(state)})
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

    stale_holds = sorted(set(quality_holds) - seen_quality_hold_ids)
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
        'source_quality_hold_receipt': quality_receipt,
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
                        help='Optional JSON receipt with excluded_items [{item_id, reason, original_source_group, split}]')
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
