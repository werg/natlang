#!/usr/bin/env python3
"""Build a hash-bound, custody-checked successor from the current broad source and verified additions.

The predecessor is streamed, not regenerated. Duplicate task IDs are reconciled, verified task treatments are
applied in place, old unordered SQL oracles are corrected, and new tasks are admitted only after checking the full
current protected union and all retained held packets. The output is immutable and retry-safe: every invocation uses
a new output directory and publishes a ready pointer only when complete.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import random
import shutil
import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds, related_components, world_identities
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file
from natlang_neuralese.common.paths import resolve
from scripts.rl_tasks.sources.sql import complexity, sql_profile


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def iter_rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def row_sha(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replay_summary(report):
    """Extract and validate the nested summary emitted by rl_tasks/verify.mjs."""
    summary = report.get('summary', {})
    if (summary.get('cases') != summary.get('kept') or summary.get('wrong_accepted', 0) or
            summary.get('wrong_errors', 0)):
        raise ValueError(f'transition replay rejected changes: {summary}')
    return summary


def sem_sha(row):
    return hashlib.sha256(json.dumps(row.get('semantics'), ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def write_rows(path: Path, items):
    with path.open('w') as f:
        for row in items:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def file_sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_identity(row):
    ext = (row.get('curriculum') or {}).get('external') or {}
    return ext.get('source', row.get('source', 'unknown'))


def _append_revision(row, revision):
    revisions = row.setdefault('source_revisions', [])
    if revision not in revisions:
        revisions.append(revision)


def _correct_ordered_sql(row):
    ext = (row.get('curriculum') or {}).get('external') or {}
    sql = ext.get('gold_sql')
    if not sql:
        return None
    profile = sql_profile(sql)
    if not profile['order_by']:
        return None
    semantics = row['semantics']
    old_oracle = semantics.get('oracle')
    if isinstance(old_oracle, str):
        old_oracle = {'level': old_oracle, 'normalization': 'result-set'}
    if not isinstance(old_oracle, dict):
        old_oracle = {'level': 'normalized', 'normalization': 'result-set'}
    if old_oracle.get('ordered') and ext.get('oracle_correction_id') == 'ordered-result-set-v1':
        return None
    before = row_sha(row)
    expected = semantics.get('expected')
    reversed_expected = list(reversed(expected)) if isinstance(expected, list) else None
    semantics['oracle'] = {**old_oracle, 'level': old_oracle.get('level', 'normalized'),
                           'normalization': old_oracle.get('normalization', 'result-set'), 'ordered': True}
    checks = ext.get('checks') or {}
    aliases = checks.get('alias_answers')
    if isinstance(aliases, list) and reversed_expected is not None and reversed_expected != expected:
        checks['alias_answers'] = [answer for answer in aliases if answer != reversed_expected]
    ext['checks'] = checks
    ext['sql_profile'] = profile
    ext['oracle_correction_id'] = 'ordered-result-set-v1'
    curriculum = row.get('curriculum') or {}
    rl = curriculum.get('rl') or {}
    rl['level'] = max(int(rl.get('level', 0) or 0), complexity(sql))
    curriculum['rl'] = rl
    row['curriculum'] = curriculum
    _append_revision(row, 'ordered-result-set-oracle@v1')
    return {'case_id': row['id'], 'old_row_sha256': before, 'new_row_sha256': row_sha(row),
            'sql_profile': profile, 'reversed_alias_removed': bool(reversed_expected is not None and
                reversed_expected != expected and reversed_expected in (aliases or []))}


def _treatment_update(old, candidate, kind):
    """Use a verified teaching treatment under the stable old task identity and world groups."""
    updated = json.loads(json.dumps(candidate))
    updated['id'] = old['id']
    updated['source_ids'] = old.get('source_ids', [])
    updated['source_groups'] = old.get('source_groups', [])
    updated['source'] = old.get('source')
    updated['gold_sources'] = old.get('gold_sources', [])
    updated['generation'] = old.get('generation', {})
    _append_revision(updated, f'{kind}@verified-guidance-v1')
    ext = (updated.get('curriculum') or {}).get('external') or {}
    ext['supersedes_case_id'] = candidate['curriculum']['external'].get('source_case_id')
    updated['curriculum']['external'] = ext
    return updated


def _restore_original_provenance(candidate, original, variant):
    row = json.loads(json.dumps(candidate))
    row['id'] = original['id']
    row['source_ids'] = original.get('source_ids', [])
    row['source_groups'] = original.get('source_groups', [])
    row['source_revisions'] = list(original.get('source_revisions', []))
    row['source'] = original.get('source')
    row['license'] = original.get('license')
    row['gold_sources'] = original.get('gold_sources', [])
    row['generation'] = original.get('generation', {})
    ext = (row.get('curriculum') or {}).get('external') or {}
    old_ext = (original.get('curriculum') or {}).get('external') or {}
    ext.update({k: old_ext[k] for k in ('upstream', 'upstream_id', 'upstream_split', 'unit') if k in old_ext})
    ext['source'] = old_ext.get('source', ext.get('source'))
    ext['source_case_id'] = original['id']
    ext['task_variant'] = variant
    row['curriculum']['external'] = ext
    _append_revision(row, f'{variant}@verified-case-derived-v1')
    return row


def inherit_versioned_assets(base_dir, additions, out, explicit=None):
    """Retain predecessor v3 assets as well as each addition's assets."""
    roots = [Path(base_dir) / 'assets-v3', *(Path(p) / 'assets-v3' for p in additions)]
    if explicit is not None:
        roots.insert(0, Path(explicit))
    roots = list(dict.fromkeys(p.resolve() for p in roots if p.is_dir()))
    if not roots:
        return None
    files = {}
    for source_root in roots:
        for source in sorted(source_root.rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root).as_posix()
            digest = sha_file(source)
            if relative in files:
                if files[relative]['sha256'] != digest:
                    raise ValueError('v3 asset collision requires an explicit task/source rewrite: ' + relative)
                continue
            target = Path(out) / 'assets-v3' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha_file(target) != digest:
                raise ValueError('v3 asset copy did not retain its exact bytes: ' + relative)
            files[relative] = {'path': relative, 'sha256': digest, 'bytes': source.stat().st_size}
    manifest = Path(out) / 'assets-v3.manifest.json'
    manifest.write_text(json.dumps({'schema': 'natlang.rl-versioned-assets-union/1',
        'source_roots': [str(p) for p in roots], 'files': list(files.values())}, indent=2) + '\n')
    return {'path': manifest.name, 'sha256': sha_file(manifest)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, type=Path)
    ap.add_argument('--base', required=True, type=Path, help='current assembled v2 broad source')
    ap.add_argument('--base-dev', required=True, type=Path)
    ap.add_argument('--base-confirmation', required=True, type=Path)
    ap.add_argument('--append', required=True, type=Path, action='append', help='assembled verified additions directory')
    ap.add_argument('--base-assets', type=Path, help='existing production task asset directory')
    ap.add_argument('--base-assets-v3', type=Path, help='explicit retained v3 assets when an older package omitted its asset tree')
    ap.add_argument('--correction-candidate', type=Path, help='four-source corrected variants package')
    ap.add_argument('--legacy-per-family', type=int, default=0)
    args = ap.parse_args()

    base_train = args.base / 'cases.jsonl' if args.base.is_dir() else args.base
    base_dir = base_train.parent
    inputs = args.append
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(f'{args.prefix}-{stamp}-{__import__("os").getpid()}')
    if out.exists():
        raise SystemExit(f'{out} already exists')
    out.mkdir(parents=True)
    (out / 'held').mkdir()
    (out / 'verification').mkdir()

    # Candidate paths and source rows are immutable; IDs stay in a small set, not full input-row maps.
    cand_train_paths = [p / 'cases.jsonl' for p in inputs]
    cand_held_paths = [(p / 'held' / 'dev.jsonl', 'dev') for p in inputs] + \
        [(p / 'held' / 'confirmation.jsonl', 'confirmation') for p in inputs]
    candidate_ids = set()
    candidate_aliases = set()
    for path in cand_train_paths:
        for row in iter_rows(path):
            candidate_ids.add(row['id'])
            ext = (row.get('curriculum') or {}).get('external') or {}
            alias = ext.get('source_case_id')
            if alias:
                candidate_aliases.add(alias)

    # Load the old holdout packets. Every existing protected receipt remains intact.
    old_dev = rows(args.base_dev)
    old_conf = rows(args.base_confirmation)
    own_old = ProtectedWorlds.from_cases({'base-dev': (file_sha(args.base_dev), old_dev),
                                          'base-confirmation': (file_sha(args.base_confirmation), old_conf)})
    protected = ProtectedWorlds.load()

    # Build collision maps only for candidate IDs and source-case aliases.
    base_ids = set()
    old_matches = {}
    base_counts = collections.Counter()
    for row in iter_rows(base_train):
        base_ids.add(row['id'])
        base_counts[_source_identity(row)] += 1
        if row['id'] in candidate_ids or row['id'] in candidate_aliases:
            old_matches[row['id']] = row

    # Candidate treatment and SQL curriculum updates are reconciled against their stable source case IDs.
    treatment_candidates = {}
    sql_tag_candidates = {}
    candidate_dup_records = []
    candidate_semantic_supersedes = {}
    candidate_training_sources = []
    for path in cand_train_paths:
        for row in iter_rows(path):
            ext = (row.get('curriculum') or {}).get('external') or {}
            source = ext.get('source')
            source_case_id = ext.get('source_case_id')
            if source in ('babi-guided', 'clutrr-guided') and source_case_id in old_matches:
                treatment_candidates[source_case_id] = (row, source)
                continue
            if source in ('spider-hard', 'bird-hard') and source_case_id in old_matches:
                sql_tag_candidates[source_case_id] = row
                candidate_dup_records.append({'case_id': row['id'], 'source_case_id': source_case_id,
                    'source': source, 'disposition': 'existing-task-curriculum-upgrade',
                    'reason': 'same upstream invocation already exists in predecessor'})
                continue
            if row['id'] in old_matches:
                old = old_matches[row['id']]
                if sem_sha(old) == sem_sha(row):
                    candidate_dup_records.append({'case_id': row['id'], 'source': source,
                        'disposition': 'duplicate-task-dropped', 'reason': 'semantics identical to predecessor'})
                    continue
                candidate_semantic_supersedes[row['id']] = row
                continue
            candidate_training_sources.append((path, row['id']))

    # Work out which new held worlds are still clean relative to the predecessor's post-withdrawal train side.
    # Baseline train is filtered against current protection and its original held packets first.
    blocked_old = protected.identities | own_old.identities
    base_train_identities = set()
    withdrawals = []
    corrections = []
    supersedes = []
    transition_changes = []
    base_spool = out / '.base-train.jsonl'
    kept_base_ids = set()
    with base_spool.open('w') as spool:
        for original in iter_rows(base_train):
            row = original
            transitioned = False
            if row['id'] in candidate_semantic_supersedes:
                candidate = candidate_semantic_supersedes[row['id']]
                old_hash = row_sha(row)
                row = json.loads(json.dumps(candidate))
                row['id'] = original['id']
                supersedes.append({'case_id': original['id'], 'old_row_sha256': old_hash,
                    'new_row_sha256': row_sha(row), 'kind': 'verified-v3-semantics-change',
                    'candidate_source': _source_identity(candidate)})
                transitioned = True
            if row['id'] in treatment_candidates:
                cand, kind = treatment_candidates[row['id']]
                before = row_sha(row)
                row = _treatment_update(row, cand, kind)
                supersedes.append({'case_id': row['id'], 'old_row_sha256': before,
                    'new_row_sha256': row_sha(row), 'kind': f'{kind}-guidance-treatment',
                    'candidate_source_case_id': cand['curriculum']['external'].get('source_case_id')})
                transitioned = True
            if row['id'] in sql_tag_candidates:
                cand = sql_tag_candidates[row['id']]
                old_profile = (cand.get('curriculum', {}).get('external') or {}).get('sql_profile')
                if old_profile:
                    ext = row['curriculum']['external']
                    ext['sql_profile'] = old_profile
                    row['curriculum']['rl']['level'] = max(int(row['curriculum']['rl'].get('level', 0) or 0),
                                                           int(cand['curriculum']['rl'].get('level', 0) or 0))
            correction = _correct_ordered_sql(row)
            if correction:
                corrections.append(correction)
                transitioned = True
            if transitioned:
                transition_changes.append(row)
            hits = sorted({kind for kind, _ in world_identities(row, evidence_items=False) & blocked_old})
            if hits:
                withdrawals.append({'case_id': row['id'], 'source': _source_identity(row),
                    'source_groups': row.get('source_groups', []), 'reason_kinds': hits,
                    'row_sha256': row_sha(row)})
                continue
            kept_base_ids.add(row['id'])
            base_train_identities.update(world_identities(row, evidence_items=False))
            spool.write(json.dumps(row, ensure_ascii=False) + '\n')

    # Collect candidate held packets, preserving all source-side held groups. A held component already exposed
    # by predecessor training is removed from the new evaluation side and receives a split-custody receipt.
    candidate_held = {'dev': [], 'confirmation': []}
    for path, split in cand_held_paths:
        if path.is_file():
            candidate_held[split].extend(rows(path))
    old_held_ids = {r['id'] for r in old_dev + old_conf}
    held_conflicts = []
    all_new_held = []
    for split, items in candidate_held.items():
        for row in items:
            if row['id'] in old_held_ids:
                candidate_dup_records.append({'case_id': row['id'], 'source': _source_identity(row),
                    'disposition': 'held-duplicate-dropped', 'reason': 'held case already exists in predecessor'})
                continue
            all_new_held.append((split, row))
    components = related_components([row for _, row in all_new_held],
                                    identities=lambda r: world_identities(r, evidence_items=False))
    keep_new_held = {'dev': [], 'confirmation': []}
    for component in components:
        members = [all_new_held[i] for i in component]
        member_splits = {split for split, _ in members}
        if len(member_splits) != 1:
            held_conflicts.append({'case_ids': [row['id'] for _, row in members],
                'source_groups': sorted({group for _, row in members for group in row.get('source_groups', [])}),
                'reason_kinds': ['cross-split-related-component'],
                'disposition': 'new-held-component-spans-dev-and-confirmation'})
            continue
        related = set().union(*(world_identities(row, evidence_items=False) for _, row in members)) & base_train_identities
        if related:
            held_conflicts.append({'case_ids': [row['id'] for _, row in members],
                'source_groups': sorted({group for _, row in members for group in row.get('source_groups', [])}),
                'reason_kinds': sorted({kind for kind, _ in related}),
                'disposition': 'new-held-component-already-exposed-in-predecessor'})
            continue
        for split, row in members:
            keep_new_held[split].append(row)
    held_rows = {'dev': old_dev + keep_new_held['dev'],
                 'confirmation': old_conf + keep_new_held['confirmation']}
    own_held = ProtectedWorlds.from_cases({split: ('pending', items) for split, items in held_rows.items()})
    blocked = protected.identities | own_held.identities

    # Add genuinely new task rows. Same-ID/same-semantics rows are duplicates, not diversity. Stable treatments
    # and SQL difficulty updates above modify the predecessor's exact rows instead of adding alternate copies.
    temp_by_source = collections.defaultdict(list)
    admitted = collections.Counter()
    excluded = []
    seen_new = set()
    for path in cand_train_paths:
        for raw in iter_rows(path):
            row = raw
            ext = (row.get('curriculum') or {}).get('external') or {}
            source = ext.get('source')
            source_case_id = ext.get('source_case_id')
            if source in ('babi-guided', 'clutrr-guided') and source_case_id in old_matches:
                continue
            if source in ('spider-hard', 'bird-hard') and source_case_id in old_matches:
                continue
            if row['id'] in old_matches:
                old = old_matches[row['id']]
                if sem_sha(old) == sem_sha(row):
                    continue
                # The verified v3 row was installed at the old ID in the predecessor pass.
                continue
            if source in ('spider-hard', 'bird-hard') and source_case_id:
                raw_source = source[:-5]
                raw_case = _load_source_case(raw_source, source_case_id)
                if raw_case:
                    row = _restore_original_provenance(row, raw_case, 'compositional-sql')
            if row['id'] in kept_base_ids or row['id'] in seen_new:
                excluded.append({'case_id': row['id'], 'source': source, 'reason': 'duplicate-task-id',
                                 'row_sha256': row_sha(row)})
                continue
            hits = sorted({kind for kind, _ in world_identities(row, evidence_items=False) & blocked})
            if hits:
                excluded.append({'case_id': row['id'], 'source': source, 'source_groups': row.get('source_groups', []),
                                 'reason': 'protected-or-held-world', 'reason_kinds': hits,
                                 'row_sha256': row_sha(row)})
                continue
            seen_new.add(row['id'])
            admitted[source or _source_identity(row)] += 1
            temp_by_source[source or _source_identity(row)].append(row)

    # Build deterministic train ordering: retain predecessor order and braid new source queues through it.
    total_new = sum(map(len, temp_by_source.values()))
    total_base = len(kept_base_ids)
    new_queues = {}
    for source, items in sorted(temp_by_source.items()):
        fams = collections.defaultdict(collections.deque)
        for row in items:
            fams[row['family']].append(row)
        ordered = []
        while fams:
            for family in sorted(list(fams)):
                ordered.append(fams[family].popleft())
                if not fams[family]:
                    del fams[family]
        new_queues[source] = collections.deque(ordered)
    new_order = []
    while new_queues:
        for source in sorted(list(new_queues)):
            new_order.append(new_queues[source].popleft())
            if not new_queues[source]:
                del new_queues[source]
    source_counts = collections.Counter()
    family_counts = collections.Counter()
    shape_counts = collections.Counter()
    instruction_hashes = collections.defaultdict(set)
    root_hashes = collections.defaultdict(set)
    levels = collections.Counter()
    helper_rows = 0
    unique_hashes = {}
    cases_path = out / 'cases.jsonl'
    base_stream = iter_rows(base_spool)
    additions_stream = iter(new_order)
    ratio = (total_new / max(1, total_base))
    acc = 0.0
    with cases_path.open('w') as f:
        def emit(row):
            nonlocal helper_rows
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            source = _source_identity(row)
            source_counts[source] += 1
            family_counts[(source, row.get('family', 'unknown'))] += 1
            ext = (row.get('curriculum') or {}).get('external') or {}
            shape_counts[(source, (row.get('curriculum') or {}).get('shape', 'unknown'))] += 1
            file = (row.get('semantics') or {}).get('files', {}).get((row.get('semantics') or {}).get('root'), '')
            body = file.split('---', 2)[-1] if isinstance(file, str) else ''
            digest = hashlib.sha256(body.encode()).hexdigest()
            instruction_hashes[source].add(digest)
            root_hashes[source].add(hashlib.sha256((row.get('semantics') or {}).get('root', '').encode()).hexdigest())
            levels[str(((row.get('curriculum') or {}).get('rl') or {}).get('level', 'unknown'))] += 1
            helper_rows += ext.get('framing') == 'neuralese-helper'
            unique_hashes[row['id']] = row_sha(row)
        for row in base_stream:
            emit(row)
            acc += ratio
            while acc >= 1 and new_order:
                new_row = next(additions_stream, None)
                if new_row is None:
                    break
                emit(new_row)
                acc -= 1
        for row in additions_stream:
            emit(row)

    # Held writes preserve the base packets and append only eligible, related-component-safe additions.
    for split, items in held_rows.items():
        write_rows(out / 'held' / f'{split}.jsonl', items)

    # Correct source revisions retain old IDs; all row changes are hash-authenticated in this sidecar.
    write_rows(out / 'supersedes.jsonl', supersedes + corrections)
    write_rows(out / 'withdrawals.jsonl', withdrawals)
    write_rows(out / 'protection-exclusions.jsonl', excluded)
    write_rows(out / 'split-conflicts.jsonl', held_conflicts)
    write_rows(out / 'duplicate-task-views.jsonl', candidate_dup_records)

    # Bind all verification reports and both asset namespaces.
    verify_rows = {}
    for label, directory in [('predecessor', base_dir / 'verification')]:
        if directory.is_dir():
            for path in directory.glob('*.json'):
                dst = out / 'verification' / f'{label}-{path.name}'
                shutil.copyfile(path, dst)
                verify_rows[str(dst.relative_to(out))] = sha_file(dst)
    for index, package in enumerate(inputs):
        directory = package / 'verification'
        if directory.is_dir():
            for path in directory.glob('*.json'):
                dst = out / 'verification' / f'append{index}-{path.name}'
                shutil.copyfile(path, dst)
                verify_rows[str(dst.relative_to(out))] = sha_file(dst)
    base_assets = args.base_assets or base_dir / 'assets'
    if not base_assets.is_dir():
        base_assets = base_dir.parent / 'natlang-rl-broad-external-20261031' / 'assets'
    if base_assets.is_dir():
        shutil.copytree(base_assets, out / 'assets')
    # Preserve any append-package assets with their original relative paths. A differing same-path blob is kept
    # under a namespaced subdirectory and explicitly recorded so runtime asset selection is never ambiguous.
    extra_asset_roots = [(p, p / 'assets') for p in inputs if (p / 'assets').is_dir()]
    asset_additions = []
    for package, asset_root in extra_asset_roots:
        package_label = package.name
        for source_file in sorted(asset_root.rglob('*')):
            if not source_file.is_file():
                continue
            relative = source_file.relative_to(asset_root)
            target = out / 'assets' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha_file(target) == sha_file(source_file):
                    continue
                target = out / 'assets' / '_append' / package_label / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                asset_additions.append({'package': str(package), 'relative_path': str(relative),
                    'output_path': str(target.relative_to(out)), 'sha256': sha_file(source_file),
                    'requires_task_specific_asset_root': True})
            else:
                asset_additions.append({'package': str(package), 'relative_path': str(relative),
                    'output_path': str(target.relative_to(out)), 'sha256': sha_file(source_file),
                    'requires_task_specific_asset_root': False})
            shutil.copyfile(source_file, target)
    asset_manifest = inherit_versioned_assets(base_dir, inputs, out, args.base_assets_v3)

    # Bind the actual predecessor, each admitted addition, protected registry, and held packets.
    candidate_receipts = []
    for package in inputs:
        rec = json.loads((package / 'receipt.json').read_text())
        q = json.loads((package / 'qualification.json').read_text())
        candidate_receipts.append({'path': str(package), 'source_id': rec['source_id'],
            'cases_sha256': rec['cases']['sha256'], 'cases_rows': rec['cases']['rows'],
            'qualification_sha256': sha_file(package / 'qualification.json'),
            'protected_sha256': q['current_protection_bindings']['protected_sha256']})
    source_id = out.name
    source_sha = sha_file(cases_path)
    qualification = {'schema': 'natlang.rl-additional-task-source-qualification/1', 'source_id': source_id,
        'source_cases_sha256': source_sha, 'qualified_task_rows': sum(source_counts.values()), 'all_train': True,
        'all_source_bound': True, 'qualification_kind': 'hash-bound successor of previously verified sources; new case rows retain upstream source IDs/groups and their source verification; SQL ordering oracles corrected in-place; guided BABI/CLUTRR cases supersede stable task IDs; no teacher generations or capability claims.',
        'training_admission': 'Parent-controlled source transition; preserve exact previous exposure and apply only to future collection/update after transition review.',
        'current_protection_intersections': {'protected_worlds': []},
        'current_protection_bindings': {'registry': REGISTRY, 'registry_sha256': sha_file(ROOT / REGISTRY),
            'protected_sha256': protected.digest, 'sources': [json.loads(x) for x in protected.sources],
            'own_held_rows': len(held_rows['dev']) + len(held_rows['confirmation'])},
        'verification': {'copied_reports': verify_rows, 'correction_cases': len(corrections),
            'replay_status': 'predecessor and addition sources individually verified; corrected ordered SQL cases retain verified expected/reference values; separate SQL oracle-correction replay is recorded as pending'},
        'source_transition': {'predecessor_cases_path': str(base_train), 'predecessor_cases_sha256': file_sha(base_train),
            'predecessor_rows': sum(base_counts.values()), 'admitted_inputs': candidate_receipts,
            'admitted_rows_by_source': dict(admitted), 'duplicate_or_treatment_records': len(candidate_dup_records),
            'superseded_rows': len(supersedes), 'oracle_corrections': len(corrections),
            'withdrawn_predecessor_rows': len(withdrawals), 'new_held_split_conflicts': len(held_conflicts),
            'protected_or_held_candidate_exclusions': len(excluded),
            'protected_source_exclusions_from_v3_build': json.loads((inputs[-1] / 'receipt.json').read_text()).get('custody_dropped', {})},
        'diversity_summary': {'rows_by_source': dict(source_counts), 'families_by_source':
            {f'{s}:{f}': n for (s, f), n in family_counts.items()}, 'shapes_by_source':
            {f'{s}:{shape}': n for (s, shape), n in shape_counts.items()},
            'distinct_root_bodies_by_source': {k: len(v) for k, v in instruction_hashes.items()},
            'distinct_root_paths_by_source': {k: len(v) for k, v in root_hashes.items()},
            'levels': dict(levels), 'neuralese_helper_rows': helper_rows,
            'semantic_quality_coverage': 'unknown; exact source verification is separate from semantic review'},
        'task_ids': list(unique_hashes), 'union_task_row_sha256': unique_hashes,
        'held': {split: {'rows': len(items), 'sha256': sha_file(out / 'held' / f'{split}.jsonl')}
                 for split, items in held_rows.items()},
        'supersedes_sha256': sha_file(out / 'supersedes.jsonl'), 'withdrawals_sha256': sha_file(out / 'withdrawals.jsonl'),
        'protection_exclusions_sha256': sha_file(out / 'protection-exclusions.jsonl'),
        'split_conflicts_sha256': sha_file(out / 'split-conflicts.jsonl'),
        'duplicate_views_sha256': sha_file(out / 'duplicate-task-views.jsonl'),
        'assets_v3_manifest': asset_manifest}
    qualification['asset_additions'] = asset_additions
    (out / 'qualification.json').write_text(json.dumps(qualification, indent=1) + '\n')
    (out / 'verification.json').write_text(json.dumps({'schema': 'natlang.broad-rl-successor-verification/1',
        'cases': sum(source_counts.values()), 'failures': [], 'exact_verification_source': 'source receipts copied above',
        'sql_order_corrections': len(corrections), 'withdrawals': len(withdrawals),
        'candidate_exclusions': len(excluded), 'held_component_conflicts': len(held_conflicts)}, indent=1) + '\n')
    receipt = {'schema': 'natlang.broad-rl-task-source/1', 'source_id': source_id, 'builder': 'scripts/rl_tasks/merge_verified_successor.py',
        'cases': {'path': 'cases.jsonl', 'sha256': source_sha, 'rows': sum(source_counts.values())},
        'held': qualification['held'], 'supersedes': {'path': 'supersedes.jsonl', 'sha256': qualification['supersedes_sha256'], 'rows': len(supersedes) + len(corrections)},
        'withdrawals': {'path': 'withdrawals.jsonl', 'sha256': qualification['withdrawals_sha256'], 'rows': len(withdrawals)},
        'protected_exclusions': {'path': 'protection-exclusions.jsonl', 'sha256': qualification['protection_exclusions_sha256'], 'rows': len(excluded)},
        'split_conflicts': {'path': 'split-conflicts.jsonl', 'sha256': qualification['split_conflicts_sha256'], 'rows': len(held_conflicts)},
        'duplicate_views': {'path': 'duplicate-task-views.jsonl', 'sha256': qualification['duplicate_views_sha256'], 'rows': len(candidate_dup_records)},
        'assets': {'base': 'assets', 'v3': 'assets-v3' if v3_dir else None}}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    # Replay every semantic treatment and oracle correction through the collector verifier before publication.
    # Newly admitted rows retain their source verifier receipts above; only changed rows need replay here.
    transition_path = out / 'transition-change-cases.jsonl'
    write_rows(transition_path, transition_changes)
    replay_dir = out / 'transition-replay'
    env = os.environ.copy()
    env.update({'NATLANG_RL_TASK_ASSETS': str(out / 'assets'),
                'NATLANG_RL_TASK_ASSETS_V3': str(out / 'assets-v3'),
                'NATLANG_HASH_CACHE': str(out / '.transition-replay-hash-cache.jsonl'),
                'NODE_OPTIONS': '--max-old-space-size=1400', 'OMP_NUM_THREADS': '1'})
    replay = subprocess.run(['node', 'scripts/rl_tasks/verify.mjs', str(ROOT / 'ts-host'),
                             str(transition_path), str(replay_dir), '4'], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    if replay.returncode:
        raise SystemExit(f'transition replay failed ({replay.returncode}): {replay.stderr[-3000:]}')
    replay_report = json.loads((replay_dir / 'verification.json').read_text())
    try:
        replay_summary_value = replay_summary(replay_report)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if replay_summary_value.get('cases') != len(transition_changes):
        raise SystemExit(f'transition replay case count mismatch: {replay_summary_value.get("cases")} != {len(transition_changes)}')
    qualification['verification']['correction_replay'] = {
        'input_sha256': sha_file(transition_path), 'report_sha256': sha_file(replay_dir / 'verification.json'),
        'cases': len(transition_changes), 'kept': replay_summary_value.get('kept'),
        'wrong_accepted': replay_summary_value.get('wrong_accepted'), 'wrong_errors': replay_summary_value.get('wrong_errors'),
        'status': 'passed'}
    qualification['verification']['replay_status'] = 'all changed rows replayed and passed; unchanged additions retain their source-level verifier receipts'
    (out / 'qualification.json').write_text(json.dumps(qualification, indent=1) + '\n')
    ready = args.prefix.with_suffix(args.prefix.suffix + '.ready.json')
    ready.write_text(json.dumps({'schema': 'natlang.retryable-assembly-ready/1', 'output': str(out),
        'qualification_sha256': sha_file(out / 'qualification.json'), 'cases_sha256': source_sha,
        'cases_rows': sum(source_counts.values()), 'sql_corrections': len(corrections)}, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'rows': sum(source_counts.values()), 'sources': len(source_counts),
        'withdrawals': len(withdrawals), 'candidate_exclusions': len(excluded), 'held_conflicts': len(held_conflicts),
        'sql_order_corrections': len(corrections), 'ready': str(ready)}, indent=2))


def _load_source_case(source, case_id):
    # The raw verified source is the source of provenance for formerly capped-out hard examples.
    base = resolve('data_nvme', 'broad-rl-tasks-20261031', 'build', source, 'verified/cases.jsonl')
    if not base.is_file():
        return None
    for row in iter_rows(base):
        if row['id'] == case_id:
            return row
    return None


def finalize_existing(out: Path):
    """Resume publication for a fully assembled directory whose transition replay finished."""
    out = out.resolve()
    cases = rows(out / 'transition-change-cases.jsonl')
    replay_dir = out / 'transition-replay'
    env = os.environ.copy()
    env.update({'NATLANG_RL_TASK_ASSETS': str(out / 'assets'),
                'NATLANG_RL_TASK_ASSETS_V3': str(out / 'assets-v3'),
                'NATLANG_HASH_CACHE': str(out / '.transition-replay-hash-cache.jsonl'),
                'NODE_OPTIONS': '--max-old-space-size=1400', 'OMP_NUM_THREADS': '1'})
    replay = subprocess.run(['node', 'scripts/rl_tasks/verify.mjs', str(ROOT / 'ts-host'),
                             str(out / 'transition-change-cases.jsonl'), str(replay_dir), '4'], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    if replay.returncode:
        raise SystemExit(f'transition replay failed ({replay.returncode}): {replay.stderr[-3000:]}')
    report_path = replay_dir / 'verification.json'
    report = json.loads(report_path.read_text())
    try:
        summary = replay_summary(report)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if summary.get('cases') != len(cases):
        raise SystemExit(f'transition replay case count mismatch: {summary.get("cases")} != {len(cases)}')
    qualification = json.loads((out / 'qualification.json').read_text())
    qualification['verification']['correction_replay'] = {
        'input_sha256': sha_file(out / 'transition-change-cases.jsonl'),
        'report_sha256': sha_file(report_path), 'cases': len(cases), 'kept': summary['kept'],
        'wrong_accepted': summary.get('wrong_accepted'), 'wrong_errors': summary.get('wrong_errors'),
        'status': 'passed'}
    qualification['verification']['replay_status'] = \
        'all changed rows replayed and passed; unchanged additions retain their source-level verifier receipts'
    (out / 'qualification.json').write_text(json.dumps(qualification, indent=1) + '\n')
    prefix = Path(str(out).rsplit('-', 2)[0])
    ready = prefix.with_suffix(prefix.suffix + '.ready.json')
    ready.write_text(json.dumps({'schema': 'natlang.retryable-assembly-ready/1', 'output': str(out),
        'qualification_sha256': sha_file(out / 'qualification.json'), 'cases_sha256': sha_file(out / 'cases.jsonl'),
        'cases_rows': sum(1 for _ in iter_rows(out / 'cases.jsonl')),
        'sql_corrections': qualification['source_transition']['oracle_corrections']}, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'ready': str(ready), 'replay': summary}, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--finalize-existing':
        finalize_existing(Path(sys.argv[2]))
        raise SystemExit(0)
    main()
