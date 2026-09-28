#!/usr/bin/env python3
"""Inventory saved static sources against a training recipe, without admitting data."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from create_training_pipeline import recipe
from run_training_pipeline import atomic_json


SOURCES = (
    'data/external_pilot/nanojev-stage1-curated-tasks.jsonl',
    'data/external_pilot/nanojev-stage1-tasks.jsonl',
    'data/external_pilot/typed-tasks.jsonl',
    'data/external_pilot/sql-tasks.jsonl',
    'data/external_pilot/sales-spread-labeled.jsonl',
    'data/external_pilot/sales-spread-topup-labeled.jsonl',
    'data/external_pilot/jeff-test-tasks.jsonl',
    'data/external_pilot/direct-core-v2.ir.jsonl',
    'data/external_pilot/sales-all-v2.ir.jsonl',
    'data/external_pilot/finqa-v1.ir.jsonl',
    'data/external_pilot/scone-v1.ir.jsonl',
    'data/external_pilot/sgd-v2.ir.jsonl',
    'data/external_pilot/clevr-v1.ir.jsonl',
    'data/external_pilot/synthetic-all-complete-702a03e8.ir.jsonl',
    'data/external_pilot/synthetic-native-reviewed-s910.ir.jsonl',
    'data/leaf_references.jsonl',
    'data/teacher/native-synthetic-s909.ir.jsonl',
    'data/teacher/legacy-decisions.archive.jsonl.gz',
    'runs/inline-curriculum/ref-v1.results.jsonl',
    'runs/inline-curriculum/ref-composed-v1.results.jsonl',
)


def inventory(path):
    counts = {name: Counter() for name in ('version', 'kind', 'split', 'family', 'gold_source', 'label_status')}
    rows = decisions = approved = errors = 0
    digest = hashlib.sha256()
    ids = set()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                errors += 1
                continue
            rows += 1
            if row.get('id'):
                ids.add(row['id'])
            for name in counts:
                value = row.get(name)
                if name == 'label_status':
                    value = row.get('jev', {}).get('status')
                counts[name][str(value)] += 1
            decisions += len(row.get('trajectory', row.get('decisions', [])))
            approved += row.get('training_admission', {}).get('approved') is True
    return {'rows': rows, 'distinct_ids': len(ids), 'parse_errors': errors,
            'uncompressed_sha256': digest.hexdigest(), 'bytes': path.stat().st_size,
            'recorded_decisions': decisions, 'approved_turns': approved,
            **{name: dict(counter) for name, counter in counts.items()}}


def audit(repo, config):
    repo = Path(repo).resolve()
    inputs = {str(Path(value.replace('${repo}', str(repo))).resolve())
              for stage in config['stages'] for value in stage.get('inputs', [])
              if '${run}' not in value}
    entries = []
    for name in SOURCES:
        path = repo / name
        if not path.exists():
            entries.append({'path': name, 'missing': True})
            continue
        entry = {'path': name, 'direct_recipe_input': str(path.resolve()) in inputs, **inventory(path)}
        if not entry['direct_recipe_input']:
            entry['disposition'] = ('evaluation_only' if set(entry['split']) <= {'test', 'dev', 'val', 'validation'}
                                    else 'not_directly_connected; inspect adapters and overlapping snapshots')
        entries.append(entry)
    return {'version': 'natlang.static_data_inventory/1', 'scope': 'explicit saved source inventory; not a completeness or quality admission claim',
            'recipe_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'entries': entries, 'excluded_legacy_turns': config.get('excluded_legacy_turns', []),
            'unit_test_corpus': config.get('unit_test_corpus', {}),
            'existing_teacher_results': config.get('existing_teacher_results', []),
            'static_source_bundle': config.get('static_source_bundle'),
            'note': 'Counts overlap across tasks/programs/snapshots; do not sum them as distinct usable examples.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.recipe.read_text()) if args.recipe else recipe(args.repo)
    result = audit(args.repo, config)
    atomic_json(args.output, result)
    print(json.dumps({'sources': len(result['entries']), 'direct_inputs': sum(entry.get('direct_recipe_input', False) for entry in result['entries'])}))
