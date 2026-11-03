#!/usr/bin/env python3
"""Register the held (dev / confirmation) rows of the authored RL pool as evaluation-only packets.

  register_authored_held.py [--pool DIR] [--stamp TAG] [--dry-run]

The authored pool (``natlang-rl-authoring/admitted/cases.jsonl``) mixes train, dev and confirmation rows. Held rows are
never selectable (the selection pool reads train rows only), but custody can only keep their *worlds* out of training
through the protected-worlds registry. Every admitted batch therefore registers its new held rows here: rows whose
ids are not yet in a registered ``natlang-rl-authored-*`` packet are written, immutably, to
``runs/natlang-rl-authored-{dev,confirmation}-<stamp>/cases.jsonl`` with a pinned manifest under
``training/corpus-manifests/`` and an ``evaluation-only-never-training`` entry in ``training/neuralese_corpora.json``.
Registered packets are never rewritten (a later batch adds a new stamped packet). The protection digest changes with each
registration; rebuild the selection pool afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

DEFAULT_POOL = resolve_str('data_hdd', 'natlang-rl-authoring', 'admitted')
REGISTRY = REPO / 'training' / 'neuralese_corpora.json'
PREFIX = 'natlang-rl-authored-'
SPLITS = {'dev': 'dev', 'confirmation': 'confirmation'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registered_ids(registry):
    ids = set()
    for entry in registry['corpora']:
        if str(entry.get('id', '')).startswith(PREFIX) and entry.get('admission') == 'evaluation-only-never-training':
            path = REPO / entry['path'] / 'cases.jsonl'
            ids |= {json.loads(line)['id'] for line in path.read_text().splitlines() if line.strip()}
    return ids


def register(pool, stamp, dry_run):
    text = REGISTRY.read_text()
    registry = json.loads(text)
    known = registered_ids(registry)
    rows = [json.loads(line) for line in (Path(pool) / 'cases.jsonl').read_text().splitlines() if line.strip()]
    pending = {split: [r for r in rows if r['split'] == split and r['id'] not in known] for split in SPLITS}
    summary = {'registered_before': len(known), 'new': {s: len(v) for s, v in pending.items()}}
    if dry_run or not any(pending.values()):
        return summary
    entries = []
    for split, items in pending.items():
        if not items:
            continue
        packet_id = f'{PREFIX}{split}-{stamp}'
        directory = REPO / 'runs' / packet_id
        if directory.exists():
            raise SystemExit(f'{directory} exists: packets are immutable; choose another --stamp')
        directory.mkdir(parents=True)
        cases = directory / 'cases.jsonl'
        cases.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
        manifest = {'schema': 'natlang.corpus-snapshot/1', 'id': packet_id, 'owner': 'dgx', 'path': f'runs/{packet_id}',
                    'files': [{'path': 'cases.jsonl', 'bytes': cases.stat().st_size, 'sha256': sha(cases),
                               'lines': len(items)}], 'bytes': cases.stat().st_size}
        (REPO / 'training' / 'corpus-manifests' / f'{packet_id}.json').write_text(json.dumps(manifest, indent=2) + '\n')
        worlds = len({r['curriculum']['external'].get('unit') for r in items})
        entries.append({'id': packet_id, 'owner': 'dgx', 'path': f'runs/{packet_id}',
                        'admission': 'evaluation-only-never-training', 'include': ['cases.jsonl'],
                        'note': f'Held {split} rows of the authored (Haiku-written, qualified) RL task pool: {len(items)} verified '
                                f'cases in {worlds} worlds (split {split}), registered {stamp}. Related invocation worlds stay on one '
                                f'split. For development and confirmation of RL outcomes; never a training source.'})
    anchor = next(i for i, e in enumerate(registry['corpora']) if e['id'] == 'natlang-rl-bgkit-kr-confirmation-20261103')
    registry['corpora'][anchor + 1:anchor + 1] = entries
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    summary['packets'] = [e['id'] for e in entries]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', default=DEFAULT_POOL)
    parser.add_argument('--stamp', default=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(register(args.pool, args.stamp, args.dry_run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
