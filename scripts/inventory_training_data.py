#!/usr/bin/env python3
"""Persistent data catalog and recipe carry-forward report; does not approve data."""
import argparse
from collections import Counter
import fnmatch
import gzip
import hashlib
import json
from pathlib import Path
import uuid

from run_training_pipeline import atomic_json


def catalog(repo, config=None):
    repo = Path(repo).resolve()
    policy_path = repo / 'training/data_sources.json'
    policy = json.loads(policy_path.read_text())
    directory = repo / 'data/teacher/data-inventory'
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / 'current.json'
    previous = json.loads(current.read_text()) if current.exists() else {'artifacts': []}
    known = {entry['path']: entry for entry in previous['artifacts']}
    inputs = set()
    if config:
        for stage in config['stages']:
            for value in stage.get('inputs', []):
                if '${run}' not in value:
                    inputs.add(str(Path(value.replace('${repo}', str(repo))).resolve()))
    artifacts = {}
    # Retain all data artifacts, not just the manually named historical inventory.
    # Group individual completed jobs separately through the snapshot's per-file ledger.
    paths = list((repo / 'data').rglob('*'))
    # Saved exports may be the only surviving copy of older runs. Inventory them
    # even when the automatic native-job snapshot cannot yet import them.
    paths += [p for p in (repo / 'runs').rglob('*')
              if p.name.endswith(('.ir.jsonl', '.results.jsonl', '.results.jsonl.gz'))]
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
                     mtime_ns=info.st_mtime_ns, direct_recipe_input=str(path) in inputs)
        entry['ever_recipe_input'] = bool(entry['direct_recipe_input'] or known.get(name, {}).get('ever_recipe_input')
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
        artifacts[name] = entry
    # Missing files remain in the catalog, rather than disappearing on the next scan.
    for name, old in known.items():
        if name not in artifacts:
            artifacts[name] = dict(old, present=False, status='missing_artifact', next_action='Locate archived/renamed source and record replacement lineage; do not silently drop.')
    snapshots = []
    snapshot_dir = repo / 'data/teacher/generated-snapshots'
    if snapshot_dir.exists():
        for path in snapshot_dir.glob('*.manifest.json'):
            value = json.loads(path.read_text())
            if not config or value['results']['path'] in inputs:
                snapshots.append(dict(value, manifest=str(path)))
    missing_inputs = []
    not_carried = []
    if config:
        for entry in artifacts.values():
            if not entry.get('ever_recipe_input') or entry.get('direct_recipe_input'):
                continue
            replacement = policy.get('replacements', {}).get(entry['path'])
            decision = next((d for d in policy['decisions'] if fnmatch.fnmatch(entry['path'], d['glob'])), None)
            resolution = ('replacement_input' if replacement and str(repo / replacement) in inputs else
                          'recorded_source_decision' if decision else
                          'caller_explicit_input_override' if config.get('data_inventory_explicit_input_override') else
                          'unreviewed_omission')
            not_carried.append({'path': entry['path'], 'resolution': resolution, 'replacement': replacement,
                                'decision': decision})
    if config and not config.get('data_inventory_explicit_input_override'):
        missing_inputs = [name for name in policy['required_default_inputs']
                          if not (repo / name).is_file() or str(repo / name) not in inputs]
        if not snapshots:
            missing_inputs.append('automatic completed-teacher snapshot')
        missing_inputs.extend(e['path'] for e in not_carried if e['resolution'] == 'unreviewed_omission')
    report = {'version': 'natlang.training_data_inventory/1', 'policy': str(policy_path),
              'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
              'artifacts': list(artifacts.values()), 'by_status': dict(Counter(e['status'] for e in artifacts.values())),
              'generated_snapshots': snapshots, 'missing_required_default_inputs': missing_inputs,
              'not_carried_forward': not_carried,
              'scope': 'Persistent data artifact catalog plus immutable generated-job snapshots with per-file admission ledgers. Counts overlap; not a final training-ready count.',
              'discovery_formats': ['jsonl', 'jsonl.gz', 'json', 'json.gz', 'parquet', 'csv', 'arrow', 'zip', 'tar.gz'],
              'unknown_data_policy': 'Visible review backlog, never silently excluded or automatically approved.'}
    report['transformations'] = [{'id': stage['id'], 'inputs': stage.get('inputs', []),
                                 'outputs': stage.get('outputs', []), 'command': stage.get('command', [])}
                                for stage in (config or {}).get('stages', [])]
    report['included_quality_blockers'] = [e['path'] for e in artifacts.values()
        if e.get('direct_recipe_input') and e['status'] == 'integrated_quality_fix_pending']
    path = directory / (str(uuid.uuid4()) + '.json')
    atomic_json(path, report)
    atomic_json(current, report)
    overview = ['# Training data inventory', '',
                'Artifact counts include derived files and archives; they are not sample counts.', '',
                '| Status | Artifacts |', '|---|---:|']
    overview += [f'| {status} | {count} |' for status, count in sorted(report['by_status'].items())]
    overview += ['', '## Generated teacher snapshot', '']
    for snapshot in snapshots:
        s = snapshot['summary']
        overview.append(f"- {s['selected_trajectories']:,} selected trajectories / {s['unique_programs']:,} unique programs from {s['completed_files']:,} completed files. Final training audit pending. Manifest: `{snapshot['manifest']}`.")
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
