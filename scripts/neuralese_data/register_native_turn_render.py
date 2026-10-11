#!/usr/bin/env python3
"""Register a finished render of render_native_turn_corpus.py (or the union) in training/neuralese_corpora.json.

Adds or replaces the registry entry ``--id`` for ``--render-dir`` (its receipt.json gives counts, tokens and hashes),
then publishes its immutable manifest (scripts/sync_training_corpora.py publish). The text files are bound per
rendering and tokenizer under ``renders`` so a recipe can name ``<path>/<rendering>/<tokenizer>/text-data.jsonl``.

Usage: register_native_turn_render.py --id ID --render-dir DIR --source-corpus ID [--notes TEXT] [--kind member|union]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'training' / 'neuralese_corpora.json'


def member_entry(a, receipt):
    renders = {}
    for key, stats in receipt['renders'].items():
        rendering, tok = key.split('/')
        renders.setdefault(rendering, {})[tok] = {
            'file': f'{rendering}/{tok}/text-data.jsonl', 'sha256': stats['text_data_sha256'],
            'documents': stats['documents'], 'by_split': stats['by_split'],
            'max_document_tokens': stats['max_document_tokens'], 'merge_omissions': stats['merge_omissions']}
    return {
        'renders': renders,
        'selection': receipt['selection'],
        'render_omissions': receipt['render_omissions'],
        'program_header_parity': receipt['program_header_parity'],
        'build': {'script': 'scripts/neuralese_data/render_native_turn_corpus.py',
                  'wrapper': 'scripts/neuralese_data/render_native_turn_corpus.sh',
                  'container': 'natlang-neuralese:26.09', 'renderer_code': receipt['renderer_code'],
                  'receipt': f'{a.path}/receipt.json', 'sources': receipt['sources'],
                  'exclude_programs': receipt['exclude_programs'], 'reserved': receipt.get('reserved'), 'protected': receipt['protected'],
                  'held_text': receipt['held_text']},
    }


def union_entry(a, receipt):
    return {'renders': {r: {tok: {'file': f'{r}/{tok}/text-data.jsonl', 'sha256': s['text_data_sha256'],
                                  'documents': s['documents'], 'by_split': s['by_split'], 'by_member': s['by_member']}
                            for tok, s in v['by_tokenizer'].items()}
                        for r, v in receipt['renderings'].items()},
            'union_omissions': {r: v['omissions'] for r, v in receipt['renderings'].items()},
            'build': {'script': 'scripts/neuralese_data/build_native_turn_union.py', 'code': receipt['code'],
                      'receipt': f'{a.path}/receipt.json'}}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    p.add_argument('--render-dir', type=Path, required=True)
    p.add_argument('--source-corpus', action='append', required=True)
    p.add_argument('--kind', choices=['member', 'union'], default='member')
    p.add_argument('--notes', default='')
    p.add_argument('--no-publish', action='store_true')
    p.add_argument('--via-worktree', action='store_true',
                   help='register, publish, commit and push through register_corpora.py (private worktree off '
                        'origin/main) instead of editing this checkout\'s registry')
    a = p.parse_args(argv)
    # The registry path is the link inside the repo, not where it resolves (corpora may live on NVMe behind a link).
    a.path = os.path.relpath(os.path.abspath(a.render_dir), ROOT)
    return a


def make_entry(a) -> dict:
    receipt = json.loads((a.render_dir / 'receipt.json').read_text())
    body = member_entry(a, receipt) if a.kind == 'member' else union_entry(a, receipt)
    entry = {
        'id': a.id, 'owner': 'dgx', 'path': a.path,
        'include': ['receipt.json', '*/*/text-data.jsonl', '*/*/omissions.jsonl', '*.jsonl', '*.json',
                    'render-omissions/*/omissions.jsonl', '*/omissions.jsonl'],
        'source_class': ('rendered-natlang-native-teacher-turn-text' if a.kind == 'member'
                         else 'shared-natlang-target-training-set'),
        'split': 'per document, inherited from the source rows; held stays held; train documents sharing a source '
                 'group with any held document (this corpus, the 4962 union, the harness bench) are dropped',
        'admission': 'renders of admitted source corpora: ' + ', '.join(a.source_corpus) +
                     '; three renderings (full: warm-up; program-header: after warm-up; whole-trajectory), LFM2.5 and '
                     'Mellum2.1 tokenizer twins; no new example admission',
        'training_admission': True,
        'training_admission_reason': 'owner 2026-10-11: maximize data; renders of training-admitted corpora '
                                     '(plans/neuralese/DATA_INVENTORY_20261011.md); program-header and whole-trajectory '
                                     'renderings are model-facing changes recorded as debt in plans/MODEL_FACING_CHANGES.md',
        'derived_from': list(a.source_corpus),
        'license': 'facts from the source rows (see the source corpora)',
        **body,
        'notes': a.notes,
    }
    return entry


def main(argv=None):
    a = parse(argv)
    entry = make_entry(a)
    if a.via_worktree:
        sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
        from register_corpora import register
        register([entry], f'Register {a.id}', replace=True)
        return
    registry = json.loads(REGISTRY.read_text())
    corpora = [c for c in registry['corpora'] if c['id'] != a.id]
    replaced = len(corpora) != len(registry['corpora'])
    corpora.append(entry)
    registry['corpora'] = corpora
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'registered': a.id, 'replaced': replaced}))
    if not a.no_publish:
        subprocess.run([sys.executable, str(ROOT / 'scripts/sync_training_corpora.py'), 'publish', '--machine', 'dgx',
                        '--id', a.id], check=True)


if __name__ == '__main__':
    main()
