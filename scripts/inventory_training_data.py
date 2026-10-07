#!/usr/bin/env python3
"""Persistent data catalog and recipe carry-forward report; does not approve data."""
import argparse
from collections import Counter
import fnmatch
import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid

from run_training_pipeline import atomic_json
from reviewed_training_inputs import resolve_reviewed_turn_inputs


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def catalog(repo, config=None):
    repo = Path(repo).resolve()
    policy_path = repo / 'training/data_sources.json'
    policy_pin = (config or {}).get('data_inventory_policy_manifest')
    if policy_pin:
        if not policy_pin.get('reason'):
            raise ValueError('Pinned inventory policy needs an explicit phase reason')
        policy_path = Path(policy_pin['path']).resolve()
        if sha256_file(policy_path) != policy_pin.get('sha256'):
            raise ValueError('Pinned inventory policy identity changed')
    policy = json.loads(policy_path.read_text())
    directory = repo / 'data/teacher/data-inventory'
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / 'current.json'
    previous = json.loads(current.read_text()) if current.exists() else {'artifacts': []}
    known = {entry['path']: entry for entry in previous['artifacts']}
    inputs = set()
    positive_carriers = set()
    if config:
        # A training pipeline may hash metadata, held source evidence, and
        # review-only preference candidates as inputs without training on them.
        # When supplied, this list is the explicitly classified positive data
        # lane used for catalog inclusion/readiness; absent the field, retain the
        # legacy behavior of treating every non-run stage input as a candidate.
        classified = config.get('data_inventory_training_inputs')
        values = classified if isinstance(classified, list) else [
            value for stage in config['stages'] for value in stage.get('inputs', [])
            if '${run}' not in value]
        for value in values:
            if '${run}' not in value:
                inputs.add(str(Path(value.replace('${repo}', str(repo))).resolve()))
        for value in config.get('data_inventory_positive_carriers', []):
            if '${run}' not in value:
                positive_carriers.add(str(Path(value.replace('${repo}', str(repo))).resolve()))
    review_inputs = set(inputs) if config and not isinstance(config.get('data_inventory_training_inputs'), list) else set()
    if config:
        for value in config.get('data_inventory_review_inputs', []):
            if '${run}' not in value:
                review_inputs.add(str(Path(value.replace('${repo}', str(repo))).resolve()))
    provenance_only = {}
    if config and config.get('data_inventory_provenance_manifest'):
        spec = config['data_inventory_provenance_manifest']
        manifest_path = Path(spec['path'].replace('${repo}', str(repo))).resolve()
        if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != spec.get('sha256'):
            raise ValueError('Provenance input manifest is missing or its pinned bytes changed')
        inventory = json.loads(manifest_path.read_text())
        if (inventory.get('schema') != 'natlang.training_existing_data_input_manifest/1' or
                inventory.get('input_count') != len(inventory.get('inputs', [])) or
                inventory.get('sha256') != hashlib.sha256(json.dumps(inventory['inputs'], sort_keys=True,
                    separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()):
            raise ValueError('Provenance input manifest failed schema or content-digest validation')
        manifest_rows = {}
        for item in inventory['inputs']:
            raw = Path(item['path'])
            path = raw if raw.is_absolute() else repo / raw
            path = path.resolve()
            if not path.is_file():
                raise ValueError(f"Provenance input is missing: {item['path']}")
            if path.stat().st_size != item.get('bytes') or sha256_file(path) != item.get('sha256'):
                raise ValueError(f"Provenance input identity changed: {item['path']}")
            if item.get('include_for_training') is False:
                if not item.get('role') or not isinstance(item.get('notes'), str) or not item['notes'].strip():
                    raise ValueError(f"Provenance input lacks an explicit role/reason: {item['path']}")
                provenance_only[str(path)] = item
            manifest_rows[str(path)] = item
        for path in inputs:
            if path not in manifest_rows or manifest_rows[path].get('include_for_training') is not True:
                raise ValueError(f'Positive training input is not classified by the pinned manifest: {path}')
        for path in positive_carriers:
            if path not in manifest_rows or manifest_rows[path].get('positive_source_carrier') is not True:
                raise ValueError(f'Positive source carrier is not classified by the pinned manifest: {path}')
    artifacts = {}
    # Retain all data artifacts, not just the manually named historical inventory.
    # Group individual completed jobs separately through the snapshot's per-file ledger.
    # Registered neuralese snapshots are catalogued by their pinned manifests;
    # do not rehash/traverse tens of GB through an unrelated native-SFT builder.
    # Their admission and pending transformations remain explicit below.
    corpus_registry_path = repo / 'training/neuralese_corpora.json'
    corpus_registry = json.loads(corpus_registry_path.read_text()) if corpus_registry_path.exists() else {'corpora': []}
    registered_corpora = []
    registered_roots = {repo / entry['path'] for entry in corpus_registry['corpora']}
    for entry in corpus_registry['corpora']:
        manifest_path = repo / 'training/corpus-manifests' / (entry['id'] + '.json')
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        registered_corpora.append({**entry, 'manifest': str(manifest_path.relative_to(repo)),
            'manifest_sha256': sha256_file(manifest_path) if manifest else None,
            'snapshot_bytes': manifest['bytes'] if manifest else None,
            'snapshot_files': len(manifest['files']) if manifest else None,
            'availability': 'present' if (repo / entry['path']).is_dir() else 'missing',
            'integration': 'dedicated_neuralese_pipeline; admission remains explicit, not a native-SFT input'})
    paths = []
    for folder, children, files in os.walk(repo / 'data'):
        children[:] = [name for name in children if Path(folder) / name not in registered_roots]
        paths.extend(Path(folder) / name for name in files)

    # Saved exports may be the only surviving copy of older runs. Inventory them
    # even when the automatic native-job snapshot cannot yet import them.
    paths += [p for p in (repo / 'runs').rglob('*')
              if p.name.endswith(('.ir.jsonl', '.results.jsonl', '.results.jsonl.gz', '.turns.jsonl', '.pairs.jsonl', '.preference-pairs.jsonl', '.corrected.jsonl'))
              or p.name in {'preference-pairs.jsonl','preferences.jsonl','pairs.jsonl','corrected.jsonl','review-candidates.jsonl'}]
    for path in sorted(paths):
        if not path.is_file() or any(part in {'data-inventory', 'generated-snapshots', 'node_modules'} for part in path.parts):
            continue
        if any(part.startswith('runtime') or part == '.git' for part in path.relative_to(repo).parts):
            continue
        if not path.name.endswith(('.jsonl', '.jsonl.gz', '.parquet', '.csv', '.arrow', '.json', '.json.gz', '.zip', '.tar.gz')):
            continue
        name = str(path.relative_to(repo))
        info = path.stat()
        entry = dict(known.get(name, {}), path=name, present=True, bytes=info.st_size,
                     mtime_ns=info.st_mtime_ns, direct_recipe_input=str(path) in inputs,
                     positive_carrier_input=str(path) in positive_carriers)
        entry['ever_recipe_input'] = bool(entry['direct_recipe_input'] or entry['positive_carrier_input'] or known.get(name, {}).get('ever_recipe_input')
                                          or known.get(name, {}).get('direct_recipe_input'))
        if known.get(name, {}).get('mtime_ns') != info.st_mtime_ns or 'observed_version' not in entry:
            entry['observed_version'] = None
            if path.name.endswith(('.jsonl', '.jsonl.gz')):
                try:
                    opener = gzip.open if path.name.endswith('.gz') else open
                    with opener(path, 'rt') as stream:
                        first = stream.readline(4_000_000)
                    row = json.loads(first)
                    entry['observed_version'] = row.get('version')
                    entry['observed_program_version'] = row.get('task', {}).get('program_ir', {}).get('version')
                    entry['first_record_sha256'] = hashlib.sha256(first.encode()).hexdigest()
                except (ValueError, UnicodeError, OSError):
                    entry['inspection_note'] = 'First record unavailable or oversize; full adapter audit required.'
        decision = next((d for d in policy['decisions'] if fnmatch.fnmatch(name, d['glob'])), None)
        if decision:
            entry.update({k: v for k, v in decision.items() if k != 'glob'})
            entry['decision_origin'] = str(policy_path.relative_to(repo))
        elif entry['direct_recipe_input']:
            entry.update(status='integrated_candidate', next_action='Apply current native, split, token and dedup gates; inclusion is not approval.')
        else:
            entry.update(status='discovered_review_pending', next_action='Identify parent source/derived artifacts, modern adapter, provenance and intended split; record a policy decision.')
        if entry.get('observed_version') == 'natlang.program/1' or entry.get('observed_program_version') == 'natlang.program/1':
            entry['required_program_version'] = 'natlang.program/2'
            entry['version_migration_pending'] = True
        if 'preference_pair' in str(entry.get('observed_version', '')) or path.name in {'preference-pairs.jsonl','preferences.jsonl','pairs.jsonl'}:
            entry['training_lane'] = 'DPO'
            entry['status'] = 'preference_review_pending'
            entry['next_action'] = 'Require current-runtime causal failure proof, approved chosen action, shared prompt and original source groups; render/audit with the student template.'
            if entry.get('observed_version') != 'natlang.preference_pair/2':
                entry['required_preference_version'] = 'natlang.preference_pair/2'
                entry['version_migration_pending'] = True
            else:
                manifest_path = Path(str(path) + '.manifest.json')
                if manifest_path.is_file():
                    pair_manifest = json.loads(manifest_path.read_text())
                    if pair_manifest.get('version') == 'natlang.preference_pair_build/1':
                        if hashlib.sha256(path.read_bytes()).hexdigest() != pair_manifest.get('pair_sha256'):
                            entry['status'] = 'preference_digest_mismatch'
                        else:
                            entry['status'] = 'preference_causally_verified_pending_render'
                            entry['verified_pairs'] = pair_manifest.get('causally_verified_pairs', 0)
                            entry['preference_sha256'] = pair_manifest['pair_sha256']
                            entry['audit_ledger'] = pair_manifest.get('audit')
                            entry['final_training_audited'] = False
        artifacts[name] = entry
    # Missing files remain in the catalog, rather than disappearing on the next scan.
    for name, old in known.items():
        if name not in artifacts:
            artifacts[name] = dict(old, present=False, status='missing_artifact', next_action='Locate archived/renamed source and record replacement lineage; do not silently drop.')
    snapshots = []
    failure_inventories = []
    snapshot_dir = repo / 'data/teacher/generated-snapshots'
    snapshot_manifests = set()
    if snapshot_dir.exists():
        snapshot_manifests.update(snapshot_dir.glob('*.manifest.json'))
    if config:
        for value in config.get('data_inventory_generated_snapshot_manifests', []):
            snapshot_manifests.add(Path(value.replace('${repo}', str(repo))).resolve())
    if snapshot_manifests:
        grouped_snapshots = {}
        grouped_failures = {}
        for path in sorted(snapshot_manifests):
            value = json.loads(path.read_text())
            if value.get('version') in {'natlang.generated_training_snapshot/1', 'natlang.generated_training_snapshot/2'}:
                if 'results' not in value or 'sha256' not in value['results']:
                    raise ValueError(f'Generated snapshot manifest lacks result identity: {path}')
                if not config or value['results']['path'] in inputs:
                    digest = value['results']['sha256']
                    grouped_snapshots.setdefault(digest, []).append(dict(value, manifest=str(path)))
            elif value.get('version') == 'natlang.teacher_failure_inventory/1':
                if 'artifact' not in value or 'sha256' not in value['artifact']:
                    raise ValueError(f'Failure inventory manifest lacks artifact identity: {path}')
                if not config or (value['artifact']['path'] in review_inputs or
                                  str(path.resolve()) in review_inputs):
                    digest = value['artifact']['sha256']
                    grouped_failures.setdefault(digest, []).append(dict(value, manifest=str(path)))
        for values in grouped_snapshots.values():
            latest = max(values, key=lambda e: e['time'])
            snapshots.append(dict(latest, equivalent_manifest_records=[e['manifest'] for e in values]))
        for values in grouped_failures.values():
            latest = max(values, key=lambda e: e['created_at'])
            failure_inventories.append(dict(latest, equivalent_manifest_records=[e['manifest'] for e in values]))
    missing_inputs = []
    required_input_resolutions = []
    not_carried = []
    if config:
        for entry in artifacts.values():
            if (not entry.get('ever_recipe_input') or entry.get('direct_recipe_input') or
                    entry.get('positive_carrier_input')):
                continue
            replacements = policy.get('replacements', {})
            chain = [entry['path']]
            while chain[-1] in replacements:
                target = replacements[chain[-1]]
                if target in chain:
                    raise ValueError('Cyclic data replacement policy: ' + ' -> '.join([*chain, target]))
                chain.append(target)
            replacement = chain[-1] if len(chain) > 1 else None
            decision = next((d for d in policy['decisions'] if fnmatch.fnmatch(entry['path'], d['glob'])), None)
            provenance = provenance_only.get(str((repo / entry['path']).resolve()))
            resolution = ('replacement_positive_input' if replacement and str((repo / replacement).resolve()) in inputs else
                          'replacement_positive_carrier' if replacement and str((repo / replacement).resolve()) in positive_carriers else
                          'recorded_source_decision' if decision else
                          'explicit_provenance_only' if provenance else
                          'caller_explicit_input_override' if config.get('data_inventory_explicit_input_override') else
                          'unreviewed_omission')
            not_carried.append({'path': entry['path'], 'resolution': resolution, 'replacement': replacement,
                                'replacement_chain': chain, 'decision': decision,
                                'provenance_role': provenance.get('role') if provenance else None,
                                'provenance_reason': provenance.get('notes') if provenance else None})
    if config and not config.get('data_inventory_explicit_input_override'):
        for name in policy['required_default_inputs']:
            original = repo / name
            decision = next((d for d in policy.get('decisions', [])
                             if d.get('glob') == name and
                             (d.get('status') == 'source_review_approved' or
                              d.get('status', '').startswith('source_review_hold'))), None)
            replacement = policy.get('replacements', {}).get(name)
            if original.is_file() and decision and replacement:
                resolved, audit, _ = resolve_reviewed_turn_inputs(repo, [original])
                if len(resolved) == 1 and resolved[0] in inputs and audit and audit[0].get('replacement') == replacement:
                    required_input_resolutions.append({'required': name, 'resolved': replacement,
                                                       'mode': 'catalog_pinned_approved_replacement',
                                                       'review_manifest': audit[0]['review_manifest'],
                                                       'review_manifest_sha256': audit[0]['review_manifest_sha256'],
                                                       'replacement_sha256': audit[0]['replacement_sha256']})
                    continue
                missing_inputs.append(name)
                continue
            if original.is_file() and str(original.resolve()) in inputs:
                required_input_resolutions.append({'required': name, 'resolved': name, 'mode': 'direct'})
                continue
            if original.is_file() and str(original.resolve()) in positive_carriers:
                required_input_resolutions.append({'required': name, 'resolved': name, 'mode': 'positive_source_carrier'})
                continue
            missing_inputs.append(name)
        if not snapshots:
            missing_inputs.append('automatic completed-teacher snapshot')
        missing_inputs.extend(e['path'] for e in not_carried if e['resolution'] == 'unreviewed_omission')
    report = {'version': 'natlang.training_data_inventory/1',
              'pinned_policy_reason': policy_pin.get('reason') if policy_pin else None, 'policy': str(policy_path),
              'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
              'artifacts': list(artifacts.values()), 'by_status': dict(Counter(e['status'] for e in artifacts.values())),
              'generated_snapshots': snapshots, 'generated_failure_inventories': failure_inventories,
              'required_input_resolutions': required_input_resolutions,
              'missing_required_default_inputs': missing_inputs,
              'explicit_provenance_only_count': len(provenance_only),
              'not_carried_forward': not_carried,
              'scope': 'Persistent data artifact catalog plus immutable generated-job snapshots with per-file admission ledgers. Counts overlap; not a final training-ready count.',
              'discovery_formats': ['jsonl', 'jsonl.gz', 'json', 'json.gz', 'parquet', 'csv', 'arrow', 'zip', 'tar.gz'],
              'unknown_data_policy': 'Visible review backlog, never silently excluded or automatically approved.'}
    report['corpus_registry'] = {'path': str(corpus_registry_path), 'snapshots': registered_corpora,
                                 'counts_overlap': True, 'admission_separate_from_replication': True}
    report['transformations'] = [{'id': stage['id'], 'inputs': stage.get('inputs', []),
                                 'outputs': stage.get('outputs', []), 'command': stage.get('command', [])}
                                for stage in (config or {}).get('stages', [])]
    report['included_quality_blockers'] = [e['path'] for e in artifacts.values()
        if (e.get('direct_recipe_input') or e.get('positive_carrier_input')) and e['status'] == 'integrated_quality_fix_pending']
    report['positive_source_carrier_count'] = sum(bool(e.get('positive_carrier_input')) for e in artifacts.values())
    report['positive_source_carriers'] = sorted(
        e['path'] for e in artifacts.values() if e.get('positive_carrier_input'))
    dpo_artifacts = [e for e in artifacts.values() if e.get('training_lane') == 'DPO']
    verified_pair_sets = {}
    for entry in dpo_artifacts:
        if entry.get('preference_sha256'):
            verified_pair_sets[entry['preference_sha256']] = entry['verified_pairs']
    report['dpo_inventory'] = {'artifacts': len(dpo_artifacts),
        'verified_native_pair_sets_by_content_hash': verified_pair_sets,
        'verified_pairs_across_distinct_content_sets': sum(verified_pair_sets.values()),
        'count_caveat': 'Different content revisions can still share pair IDs; final DPO preparation must dedup IDs/rendered pairs and isolate source groups.',
        'final_training_audited': False}
    path = directory / (str(uuid.uuid4()) + '.json')
    atomic_json(path, report)
    atomic_json(current, report)
    overview = ['# Training data inventory', '',
                'Artifact counts include derived files and archives; they are not sample counts.', '',
                '| Status | Artifacts |', '|---|---:|']
    overview += [f'| {status} | {count} |' for status, count in sorted(report['by_status'].items())]
    overview += ['', '## Registered neuralese snapshots', '']
    for corpus in registered_corpora:
        overview.append(f"- `{corpus['id']}`: {corpus['availability']}; {corpus['admission']}; manifest `{corpus['manifest']}`.")
    overview += ['', '## Generated teacher snapshot', '']
    for snapshot in snapshots:
        s = snapshot['summary']
        overview.append(f"- {s['selected_trajectories']:,} selected trajectories / {s['unique_programs']:,} unique programs from {s['completed_files']:,} completed files. Final training audit pending. Manifest: `{snapshot['manifest']}`.")
        if s.get('exclusion_categories'):
            overview.append('- Held-file disposition counts (can overlap): ' + ', '.join(f'{k}: {v:,}' for k,v in s['exclusion_categories'].items()))
    overview += ['', '## Generated teacher failure candidates', '']
    if failure_inventories:
        for inventory in failure_inventories:
            overview.append(f"- {inventory['candidates']:,} retained failure candidates; immutable artifact `{inventory['artifact']['path']}` (SHA-256 `{inventory['artifact']['sha256']}`), manifest `{inventory['manifest']}`. Candidate retention does not assign DPO negatives or create pairs.")
    else:
        overview.append('- No generated failure-candidate inventory was carried into this recipe.')
    overview += ['', '## DPO pairs', '',
                 f"{report['dpo_inventory']['artifacts']} recorded pair artifacts; {report['dpo_inventory']['verified_pairs_across_distinct_content_sets']} native causally verified pairs after collapsing byte-identical sets. Final rendering/split/token/dedup audit pending.",
                 report['dpo_inventory']['count_caveat']]
    overview += ['', '## Transformation and review work', '', '| Source pattern | Status | Next action |', '|---|---|---|']
    overview += [f"| `{d['glob']}` | {d['status']} | {d.get('next_action', d.get('reason', ''))} |" for d in policy['decisions']]
    overview += ['', '## Carry-forward and training blockers', '',
                 f"Missing/unreviewed omissions: {len(missing_inputs)}.", '',
                 *[f'- `{name}`' for name in missing_inputs + report['included_quality_blockers']], '',
                 f'Full immutable report: `{path}`.', 'Persistent catalog: `current.json`.',
                 'Per-file generated-job ledgers explain every admission hold and omitted duplicate.']
    (directory / 'overview.md').write_text('\n'.join(overview) + '\n')
    if missing_inputs:
        raise ValueError('Training inputs lost or missing; restore or explicitly override: ' + ', '.join(missing_inputs))
    return path, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--input', action='append', default=[])
    parser.add_argument('--allow-input-override', action='store_true')
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--ready-out', type=Path)
    args = parser.parse_args()
    if args.check_report:
        report = json.loads(args.check_report.read_text())
        blockers = report.get('missing_required_default_inputs', []) + report.get('included_quality_blockers', [])
        if blockers:
            raise SystemExit('Training data inventory blocked: ' + ', '.join(blockers))
        if not args.ready_out:
            parser.error('--check-report requires --ready-out')
        atomic_json(args.ready_out, {'ready': True, 'report': str(args.check_report),
                                    'sha256': hashlib.sha256(args.check_report.read_bytes()).hexdigest()})
        raise SystemExit(0)
    config = json.loads(args.recipe.read_text()) if args.recipe else None
    if args.input:
        if config:
            parser.error('choose recipe or explicit inputs')
        config = {'stages': [{'inputs': [str(Path(p).resolve()) for p in args.input]}],
                  'data_inventory_explicit_input_override': args.allow_input_override}
    path, report = catalog(args.repo, config)
    print(json.dumps({'report': str(path), 'by_status': report['by_status'], 'missing_required_default_inputs': report['missing_required_default_inputs']}))
