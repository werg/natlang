#!/usr/bin/env python3
"""Acquire/prepare/audit/register artifact sources; never launch ungraded collection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from acquire_visual_sources import digest, write_json
from audit_visual_source_tasks import audit
from prepare_visual_source_tasks import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('training/visual_sources.json'))
    parser.add_argument('--task-registry', type=Path, default=Path('training/self_improvement_tasks.json'))
    parser.add_argument('--raw', type=Path, default=Path('vendor/datasets/visual-frontend'))
    parser.add_argument('--out', type=Path, default=Path('data/self-improvement/visual-frontend/intake-v3'))
    parser.add_argument('--source', default='all')
    parser.add_argument('--acquire', action='store_true')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    selected = set(args.source.split(','))
    sources = [s for s in registry['sources'] if args.source == 'all' or s['id'] in selected]
    if args.source != 'all' and selected != {s['id'] for s in sources}:
        parser.error('unknown source')
    if args.acquire:
        subprocess.run([sys.executable, str(Path(__file__).with_name('acquire_visual_sources.py')),
            '--registry', str(args.registry), '--out', str(args.raw), '--source', args.source], check=True)
    tasks_registry = json.loads(args.task_registry.read_text())
    entries = {row['id']: row for row in tasks_registry.get('artifact_sources', [])}
    reports, errors = [], []
    for source in sources:
        try:
            directory = args.out / source['id']
            if not directory.exists():
                prepare(source, args.raw, args.out)
            report = audit(directory)
            if not report['passed']:
                raise ValueError('; '.join(report['errors']))
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest['source'] != source:
                raise ValueError('prepared source lock differs')
            receipt = args.raw / source['id'] / source['revision'] / 'acquisition.json'
            if digest(receipt) != manifest['acquisition_receipt_sha256']:
                raise ValueError('acquisition receipt differs from preparation')
            write_json(directory / 'audit.json', report)
            previous = entries.get('visual-' + source['id'])
            if previous and previous['manifest'] != str(directory / 'manifest.json'):
                history = tasks_registry.setdefault('artifact_source_history', [])
                if not any(h['manifest'] == previous['manifest'] for h in history):
                    history.append({**previous, 'state': 'superseded_source_preparation', 'replacement': str(directory / 'manifest.json'), 'reason': 'New immutable adapter preparation; preserve former artifacts and audits as history.'})
            entries['visual-' + source['id']] = {'id': 'visual-' + source['id'],
                'state': 'source_prepared_executor_pending', 'source_id': source['id'],
                'tasks': str(directory / 'artifact-source-tasks.jsonl'), 'sha256': manifest['sha256'],
                'manifest': str(directory / 'manifest.json'), 'manifest_sha256': digest(directory / 'manifest.json'),
                'audit': str(directory / 'audit.json'), 'audit_sha256': digest(directory / 'audit.json'),
                'acquisition': str(receipt), 'acquisition_sha256': digest(receipt),
                'source_registry': str(args.registry), 'source_registry_sha256': digest(args.registry),
                'next_action': 'Global source-group split review; independent artifact evaluator; resolve each row blocker before constructing collector SkillEpisodes.',
                'training_admitted': False}
            reports.append({'source': source['id'], 'tasks': manifest['tasks'], 'groups': manifest['groups'], 'statuses': manifest['statuses']})
        except (OSError, ValueError, KeyError) as error:
            errors.append({'source': source['id'], 'error': str(error)})
    tasks_registry['artifact_sources'] = list(entries.values())
    # Existing native corpora, collections and recovery plans remain in their own lanes.
    write_json(args.task_registry, tasks_registry)
    summary = {'schema': 'natlang.visual-source-intake/1', 'sources': reports, 'errors': errors,
        'training_admitted': False, 'provider_calls': 0, 'source_registry_sha256': digest(args.registry)}
    write_json(args.out / 'intake.json', summary)
    print(json.dumps(summary, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
