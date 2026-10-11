#!/usr/bin/env python3
"""Machine-readable training-data inventory (plans/neuralese/DATA_INVENTORY_20261011.md).

Reads the corpus registry and the immutable manifests and classifies every registered corpus as training-shaped,
generation source, evidence, or model/training state, with its size from the manifest. It then attaches the curated
2026-10-11 audit: the ranked datasets with origin, format, size, admission decision and conversion; the defect sets
for Pop to repair; the unregistered on-disk datasets; and the measured direct-stage budget.

Usage: python3 scripts/neuralese_data/build_data_inventory.py [--out plans/neuralese/DATA_INVENTORY_20261011.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CURATED = REPO / 'plans/neuralese/DATA_INVENTORY_20261011.curated.json'

MODEL_STATE_WORDS = ('checkpoint', 'model-artifact', 'full-state', 'fullstate', 'state-only', 'weights',
                     'training-state', 'closed-run', 'retained-model', 'resumable-state', 'model-diagnostic',
                     'teacher-model-artifact', 'heads-only', 'best-state')
SOURCE_WORDS = ('source-only', 'source-tasks', 'source-episodes', 'generation-source', 'generation-only',
                'generation-input', 'verified-source-cases', 'source/reference', 'source-view')


def classify(entry):
    text = json.dumps(entry.get('admission')).lower()
    if entry.get('training_admission') is True:
        return 'training-admitted'
    if any(word in text for word in MODEL_STATE_WORDS):
        return 'model-or-training-state'
    if 'protected' in text and 'never' in text:
        return 'protected-evaluation'
    if any(word in text for word in SOURCE_WORDS):
        return 'generation-source'
    return 'held-or-evidence'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO / 'plans/neuralese/DATA_INVENTORY_20261011.json')
    a = parser.parse_args(argv)
    registry = json.loads((REPO / 'training/neuralese_corpora.json').read_text())
    curated = json.loads(CURATED.read_text())
    decided = {row['id']: row for row in curated['datasets'] if 'id' in row}
    rows = []
    for entry in registry['corpora']:
        manifest_path = REPO / 'training/corpus-manifests' / (entry['id'] + '.json')
        files = json.loads(manifest_path.read_text()).get('files', []) if manifest_path.is_file() else []
        rows.append({
            'id': entry['id'], 'owner': entry['owner'], 'path': entry['path'],
            'class': classify(entry),
            'bytes': sum(f.get('bytes', 0) for f in files),
            'jsonl_lines': sum(f.get('lines', 0) or 0 for f in files if f['path'].endswith('.jsonl')),
            'admission': entry.get('admission'), 'training_admission': entry.get('training_admission'),
            'derived_from': entry.get('derived_from'),
            'audit_2026_10_11': decided.get(entry['id'], {}).get('decision'),
        })
    counts = {}
    for row in rows:
        bucket = counts.setdefault(row['class'], {'corpora': 0, 'bytes': 0, 'jsonl_lines': 0})
        bucket['corpora'] += 1
        bucket['bytes'] += row['bytes']
        bucket['jsonl_lines'] += row['jsonl_lines']
    out = {'schema': 'natlang.training-data-inventory/1', 'date': '2026-10-11',
           'doc': 'plans/neuralese/DATA_INVENTORY_20261011.md',
           'registry_counts_by_class': counts, **curated, 'registry': rows}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'registry_rows': len(rows), 'classes': counts}))


if __name__ == '__main__':
    main()
