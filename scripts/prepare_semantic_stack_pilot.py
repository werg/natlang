#!/usr/bin/env python3
"""Seal one-case Luna queues from a pinned semantic-stack priority selection.

The authored priority plan decides the order. This script only copies the first
N selected Program/2 rows and binds queue indices, runtime, and hashes. It
does not launch workers or admit training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed-base', type=int, default=782300)
    parser.add_argument('--max-turns', type=int, default=600)
    parser.add_argument('--case-seconds', type=int, default=14400)
    args = parser.parse_args()
    selection_path = args.selection.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('schema') != 'natlang.semantic-stack-priority-selection/1':
        raise ValueError('Selection must be a sealed semantic-stack priority manifest')
    if selection.get('training_admission') is not False or selection.get('teacher_observations_credited') != 0:
        raise ValueError('Selection must be preparation-only')
    source = Path(selection['selected_source']).resolve(strict=True)
    if digest(source) != selection['selected_source_sha256']:
        raise ValueError('Selected source bytes changed')
    if args.count < 1 or args.count > len(selection['selected_ids']):
        raise ValueError('Count exceeds selected source')
    runtime = args.runtime.resolve(strict=True)
    runtime_manifest = runtime / 'frozen-runtime.json'
    if not runtime_manifest.is_file():
        raise ValueError('Runtime must have frozen-runtime.json')
    # Reuse the existing exact runtime-tree verifier; no mutable checkout code
    # is executed by a pilot worker.
    from freeze_training_runtime import tree_identity
    frozen = json.loads(runtime_manifest.read_text(encoding='utf-8'))
    if tree_identity(runtime) != frozen['files']:
        raise ValueError('Frozen runtime tree changed')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['id'] for row in rows] != selection['selected_ids']:
        raise ValueError('Priority source/order mismatch')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pilot_source = out / 'pilot-source.ir.jsonl'
    pilot_source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[:args.count]), encoding='utf-8')
    queues = []
    for index, row in enumerate(rows[:args.count]):
        worker = out / f'case-{index:03d}'
        worker.mkdir()
        case = worker / 'collection'
        entry = {
            'action_max_tokens': 4096,
            'case_seconds': args.case_seconds,
            'context_tokens': 32768,
            'count': 1,
            'index': index,
            'jobs': str(case / 'jobs'),
            'key': f"semstack-priority-{hashlib.sha256(row['id'].encode()).hexdigest()[:16]}-seed{args.seed_base + index}",
            'log': str(case / 'collector.log'),
            'max_model_requests': 4096,
            'max_turns': args.max_turns,
            'output': str(case / 'results.jsonl'),
            'request_retries': 3,
            'seed': args.seed_base + index,
            'source': str(pilot_source),
            'text_neuralese_emulation': True,
            'transport_retries': 0,
        }
        queue = worker / 'queue.jsonl'
        queue.write_text(json.dumps(entry, sort_keys=True) + '\n', encoding='utf-8')
        queues.append({'id': row['id'], 'queue': str(queue), 'queue_sha256': digest(queue)})
    manifest = {
        'schema': 'natlang.semantic-stack-priority-pilot/1',
        'selection': str(selection_path),
        'selection_sha256': digest(selection_path),
        'parent_source_sha256': digest(source),
        'source': str(pilot_source),
        'source_sha256': digest(pilot_source),
        'runtime': str(runtime),
        'runtime_manifest_sha256': digest(runtime_manifest),
        'queues': queues,
        'resource_ceiling': {'case_seconds': args.case_seconds, 'max_turns': args.max_turns,
                             'max_model_requests': 4096, 'context_tokens': 32768},
        'teacher_observations_credited': 0,
        'training_admission': False,
        'review': 'Diagnostic collection only. Independently inspect final semantics, source witnesses, actual NL call/child-result use, and idiomaticity before row admission.',
    }
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(manifest_path), 'source_sha256': manifest['source_sha256'],
                      'queues': queues}, indent=2))


if __name__ == '__main__':
    main()
