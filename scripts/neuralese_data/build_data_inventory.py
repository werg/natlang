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


def _first_record(path: Path):
    """Sample JSONL, or a bounded JSON document; never load an entire corpus."""
    try:
        with path.open('r', encoding='utf-8') as stream:
            if path.suffix.lower() == '.json':
                # Pretty-printed manifests are common. Bound metadata reads to
                # avoid loading large array exports during an inventory sweep.
                document = stream.read(1_048_577)
                if len(document) > 1_048_576:
                    return None
                value = json.loads(document)
                return value if isinstance(value, (dict, list)) else None
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return None


def _semantic_record_shape(relative_path, value):
    """Classify data shape independently from admission or source quality.

    The inventory is a routing aid for the bulk renderer. It recognizes rows
    with explicit native targets, rendered-token supervision, or source/task
    inputs that may need conversion. It does not infer correctness,
    split eligibility, or admission from filenames or JSON keys.
    """
    if value is None:
        return None
    keys = set(value)
    version = value.get('version') or value.get('schema')
    version = version if isinstance(version, str) else None
    if version in {'natlang.teacher_training_turn.native/1', 'natlang.teacher_trajectory.native/1'}:
        return 'training-shaped'
    if version == 'natlang.port-record/1':
        return 'training-shaped'
    if version == 'natlang.teacher_partial.native/1':
        return 'conversion-input'
    if (('token_ids' in keys and any(k in keys for k in ('supervised_suffix_start', 'target_start'))) or
            ('input_ids' in keys and 'labels' in keys)):
        return 'training-shaped'
    if (('messages' in keys and any(k in keys for k in ('target', 'target_text', 'assistant_target'))) or
            ('prompt_ids' in keys and 'response_ids' in keys)):
        return 'training-shaped'
    if 'task' in keys and ('provisional_gold' in keys or 'target' in keys or 'program_id' in keys):
        return 'training-shaped'
    if ('trajectory' in keys and ('outcome' in keys or 'target' in keys)):
        return 'conversion-input'
    if (keys & {'question', 'options', 'choices', 'prompt', 'dataset_records'} or
            ('state' in keys and keys & {'task', 'label', 'expected', 'gold'}) or
            ('label' in keys and keys & {'context', 'question', 'candidate', 'gold'})):
        return 'conversion-input'
    return None


def classify_shape(entry, manifest_files):
    """Return observed row shape without converting it into training authority.

    Classification is file-level and conservative: sample one row from each
    declared JSON/JSONL member, recognize known trajectory/rendered/source
    schemas, and leave everything else unknown/evidence. Mixed snapshots remain
    mixed so their review and audit files cannot hide their data members.
    """
    root = REPO / entry['path']
    shapes = {}
    examples = {}
    member_totals = {}
    for item in manifest_files:
        rel = item['path']
        path = root / rel
        suffix = Path(rel).suffix.lower()
        shape = None
        if suffix in {'.json', '.jsonl'}:
            value = _first_record(path)
            if isinstance(value, list):
                value = value[0] if value and isinstance(value[0], dict) else None
            if suffix == '.json' and value is not None:
                # For JSON arrays/wrappers, inspect only the first declared row.
                for key in ('rows', 'records', 'items', 'cases', 'examples'):
                    nested = value.get(key)
                    if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                        value = nested[0]
                        break
            shape = _semantic_record_shape(rel, value)
        if shape is None:
            lowered = rel.lower()
            if suffix in {'.pt', '.pth', '.safetensors', '.gguf', '.bin'}:
                shape = 'model-or-training-state'
            elif suffix in {'.log', '.md', '.sha256'} or any(word in lowered for word in
                    ('review', 'audit', 'receipt', 'approval', 'launch', 'summary', 'report', 'metric', 'trace', 'verification')):
                shape = 'evidence'
            else:
                shape = 'unclassified'
        shapes[shape] = shapes.get(shape, 0) + 1
        totals = member_totals.setdefault(shape, {'bytes': 0, 'jsonl_lines': 0})
        totals['bytes'] += item.get('bytes', 0)
        if rel.endswith('.jsonl'):
            totals['jsonl_lines'] += item.get('lines', 0) or 0
        examples.setdefault(shape, [])
        if len(examples[shape]) < 3:
            examples[shape].append(rel)
    observed_data = shapes.get('training-shaped', 0) + shapes.get('conversion-input', 0)
    observed_evidence = shapes.get('evidence', 0)
    if observed_data and observed_evidence:
        summary = 'mixed'
    elif shapes.get('training-shaped'):
        summary = 'training-shaped'
    elif shapes.get('conversion-input'):
        summary = 'conversion-input'
    elif observed_evidence and not shapes.get('unclassified'):
        summary = 'pure-evidence'
    else:
        summary = 'unclassified'
    return {'class': summary, 'member_counts': shapes, 'member_totals': member_totals, 'examples': examples,
            'note': 'Structural sample only; does not establish quality, visibility, split, or training admission.'}


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
        shape = classify_shape(entry, files)
        rows.append({
            'id': entry['id'], 'owner': entry['owner'], 'path': entry['path'],
            'class': classify(entry),
            'data_shape': shape,
            'bytes': sum(f.get('bytes', 0) for f in files),
            'jsonl_lines': sum(f.get('lines', 0) or 0 for f in files if f['path'].endswith('.jsonl')),
            'admission': entry.get('admission'), 'training_admission': entry.get('training_admission'),
            'derived_from': entry.get('derived_from'),
            'audit_2026_10_11': decided.get(entry['id'], {}).get('decision'),
        })
    counts = {}
    shape_counts = {}
    for row in rows:
        bucket = counts.setdefault(row['class'], {'corpora': 0, 'bytes': 0, 'jsonl_lines': 0})
        bucket['corpora'] += 1
        bucket['bytes'] += row['bytes']
        bucket['jsonl_lines'] += row['jsonl_lines']
        shape = row['data_shape']['class']
        shape_bucket = shape_counts.setdefault(shape, {'corpora': 0, 'bytes': 0, 'jsonl_lines': 0})
        shape_bucket['corpora'] += 1
        shape_bucket['bytes'] += row['bytes']
        shape_bucket['jsonl_lines'] += row['jsonl_lines']
    out = {'schema': 'natlang.training-data-inventory/1', 'date': '2026-10-11',
           'doc': 'plans/neuralese/DATA_INVENTORY_20261011.md',
           'registry_counts_by_class': counts,
           'registry_counts_by_data_shape': shape_counts,
           **curated, 'registry': rows}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'registry_rows': len(rows), 'classes': counts}))


if __name__ == '__main__':
    main()
