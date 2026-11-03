#!/usr/bin/env python3
"""Register the held (dev / confirmation) packets of an assembled RL task source as evaluation-only.

  register_source_held.py SOURCE_DIR --id natlang-rl-bgkit-kr --stamp 20261104 --label "bgkit knowledge-reasoning RL reader source" [--dry-run]

SOURCE_DIR is an assembled source (``scripts/rl_tasks/assemble.py``) with ``held/dev.jsonl`` and ``held/confirmation.jsonl``.
Each is copied, immutably, to ``runs/<id>-{dev,confirmation}-<stamp>/cases.jsonl`` with a pinned manifest under
``training/corpus-manifests/`` and an ``evaluation-only-never-training`` entry in ``training/neuralese_corpora.json`` (after
the entries of the same id prefix), so custody keeps their worlds out of every training pool. A registered packet is never
rewritten: a later rebuild registers a new stamped packet. The protection digest changes; rebuild selection pools afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha  # noqa: E402

REGISTRY = REPO / 'training' / 'neuralese_corpora.json'
SPLITS = {'dev': 'dev.jsonl', 'confirmation': 'confirmation.jsonl'}


def register(source, prefix, stamp, label, dry_run):
    registry = json.loads(REGISTRY.read_text())
    ids = {entry['id'] for entry in registry['corpora']}
    entries, summary = [], {}
    for split, name in SPLITS.items():
        packet_id = f'{prefix}-{split}-{stamp}'
        rows = [line for line in (Path(source) / 'held' / name).read_text().splitlines() if line.strip()]
        summary[packet_id] = len(rows)
        if packet_id in ids or (REPO / 'runs' / packet_id).exists():
            raise SystemExit(f'{packet_id} is already registered: packets are immutable; choose another --stamp')
        if dry_run:
            continue
        directory = REPO / 'runs' / packet_id
        directory.mkdir(parents=True)
        cases = directory / 'cases.jsonl'
        cases.write_text(''.join(line + '\n' for line in rows))
        manifest = {'schema': 'natlang.corpus-snapshot/1', 'id': packet_id, 'owner': 'dgx', 'path': f'runs/{packet_id}',
                    'files': [{'path': 'cases.jsonl', 'bytes': cases.stat().st_size, 'sha256': sha(cases), 'lines': len(rows)}],
                    'bytes': cases.stat().st_size}
        (REPO / 'training' / 'corpus-manifests' / f'{packet_id}.json').write_text(json.dumps(manifest, indent=2) + '\n')
        entries.append({'id': packet_id, 'owner': 'dgx', 'path': f'runs/{packet_id}',
                        'admission': 'evaluation-only-never-training', 'include': ['cases.jsonl'],
                        'note': f'Held {split} packet of the {label} ({len(rows)} verified cases, split {split}), registered '
                                f'{stamp}. Related units share a split. For development and confirmation of RL outcomes; '
                                f'never a training source.'})
    if entries:
        anchor = max(i for i, entry in enumerate(registry['corpora']) if entry['id'].startswith(prefix + '-'))
        registry['corpora'][anchor + 1:anchor + 1] = entries
        REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source')
    parser.add_argument('--id', required=True, help='packet id prefix, e.g. natlang-rl-bgkit-kr')
    parser.add_argument('--stamp', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(register(args.source, args.id, args.stamp, args.label, args.dry_run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
