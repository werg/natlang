#!/usr/bin/env python3
"""Freeze the typed-label pool and its shared persistence dependency as one artifact."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'scripts/run_gemini_decision_pool.py',
    'scripts/label_decision_cases.py',
    'training/neuralese/natlang_neuralese/__init__.py',
    'training/neuralese/natlang_neuralese/common/__init__.py',
    'training/neuralese/natlang_neuralese/common/jsonio.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--models', type=Path,
                        default=ROOT / 'training/providers/gemini-free-text-pool.json')
    args = parser.parse_args()
    models = json.loads(args.models.read_text())
    if not models.get('models'):
        raise ValueError('A declared model pool is required')
    args.destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for source, name in [(ROOT / name, name) for name in FILES] + [(args.models, 'models.json')]:
        data = source.read_bytes()
        target = args.destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append({'path': name, 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'schema': 'natlang.frozen-decision-pool/1',
                'entrypoint': 'scripts/run_gemini_decision_pool.py',
                'files': entries, 'training_admission': False}
    (args.destination / 'frozen-runtime.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'destination': str(args.destination), 'files': len(entries),
                      'bytes': sum(entry['bytes'] for entry in entries)}))


if __name__ == '__main__':
    main()
