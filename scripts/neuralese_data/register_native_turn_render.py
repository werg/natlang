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
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))


def verify_union_protection(render_dir, receipt):
    """Require current build bindings or a current, byte-bound independent audit."""
    from source_review import registry_sha256
    from neuralese_data.decision_holds import load_decision_holds
    from natlang_neuralese.common.hashing import sha256_file_hex
    review_dir = ROOT / 'training/source-reviews'
    hold_dir = ROOT / 'training/decision-holds'
    hold_paths = sorted(hold_dir.glob('*.jsonl'))
    if any(not any(p.name.startswith(prefix) for p in hold_paths)
           for prefix in ('semantic-ticket-crisp-bypass-', 'native-non-success-')):
        raise ValueError('union admission requires current exact decision-hold files')
    review_sha = registry_sha256(review_dir)
    held_ids, bindings = load_decision_holds(hold_paths)

    def identities(items):
        return {(Path(b['path']).name, b['sha256']) for b in items}

    expected = identities(bindings)
    current_build = ((receipt.get('source_review') or {}).get('holds_sha256') == review_sha and
                     expected <= identities(b for b in receipt.get('excluded_source_references', [])
                                            if b.get('kind') == 'decision-hold') and
                     receipt.get('held_native_decision_ids', -1) >= len(held_ids))
    proof_binding = None
    if not current_build:
        path = render_dir / 'independent-protection-audit.json'
        if not path.is_file():
            raise ValueError('union admission requires current source-review and exact decision-hold bindings')
        proof = json.loads(path.read_text())
        if not (proof.get('schema') == 'natlang.protected-union-independent-audit/2' and
                proof.get('status') == 'passed' and proof.get('current_source_review_sha256') == review_sha and
                expected <= identities(proof.get('current_exact_hold_files', [])) and
                proof.get('twin_identities_equal') is True and
                all(proof.get(key) == 0 for key in ('pending_source_review_intersections',
                    'exact_decision_hold_intersections', 'train_held_group_text_intersections',
                    'quality_quarantine_intersections'))):
            raise ValueError('union independent protection audit is stale or incomplete')
        outputs = {f'{mode}:{tok}': render_dir / mode / tok / stats.get('text_data_file', 'text-data.jsonl')
                   for mode, view in receipt['renderings'].items()
                   for tok, stats in view['by_tokenizer'].items()}
        if set(proof.get('files', {})) != set(outputs):
            raise ValueError('union protection audit does not cover every output view')
        for key, output in outputs.items():
            if sha256_file_hex(output) != proof['files'][key].get('sha256'):
                raise ValueError('union output changed after protection audit: ' + key)
        proof_binding = {'path': str(path), 'sha256': sha256_file_hex(path)}
    # Reviews can advance while large output files are being hashed.
    _, after = load_decision_holds(sorted(hold_dir.glob('*.jsonl')))
    if registry_sha256(review_dir) != review_sha or identities(after) != expected:
        raise ValueError('union protection inputs changed during registration; re-audit current holds')
    return {'source_review_sha256': review_sha, 'exact_decision_hold_bindings': bindings,
            'independent_audit': proof_binding, 'current_build_receipt': current_build}


def text_binding(rendering, tokenizer, stats):
    name = stats.get('text_data_file', 'text-data.jsonl')
    if name not in ('text-data.jsonl', 'text-data.jsonl.zst'):
        raise ValueError('unsupported rendered text filename: ' + str(name))
    binding = {'file': f'{rendering}/{tokenizer}/{name}', 'sha256': stats['text_data_sha256']}
    if name.endswith('.zst'):
        import re
        digest, size = stats.get('decoded_text_data_sha256'), stats.get('decoded_text_data_bytes')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or type(size) is not int or size < 0:
            raise ValueError('compressed text requires verified decoded hash and byte count')
        binding.update(compression='zstd', decoded_sha256=digest, decoded_bytes=size)
    return binding


def member_entry(a, receipt):
    renders = {}
    for key, stats in receipt['renders'].items():
        rendering, tok = key.split('/')
        renders.setdefault(rendering, {})[tok] = {
            **text_binding(rendering, tok, stats),
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
    return {'renders': {r: {tok: {**text_binding(r, tok, s),
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
    p.add_argument('--machine', choices=['pop', 'dgx'], default='dgx',
                   help='machine that owns and publishes the render artifacts')
    p.add_argument('--notes', default='')
    p.add_argument('--supersedes', action='append', default=[], help='registered ids this render replaces')
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
    protection = verify_union_protection(a.render_dir, receipt) if a.kind == 'union' else None
    body = member_entry(a, receipt) if a.kind == 'member' else union_entry(a, receipt)
    if protection is not None:
        body['build']['admission_protection'] = protection
    sources = {corpus['id']: corpus for corpus in json.loads(REGISTRY.read_text())['corpora']}
    unadmitted = [source for source in a.source_corpus
                  if source not in sources or sources[source].get('training_admission') is not True]
    if unadmitted:
        raise ValueError(f'render sources lack explicit training admission: {unadmitted}')
    variants = ', '.join(f'{rendering}/{tokenizer}' for rendering, tokenizers in body['renders'].items()
                         for tokenizer in tokenizers)
    entry = {
        'id': a.id, 'owner': a.machine, 'path': a.path,
        'include': ['receipt.json', '*/*/text-data.jsonl', '*/*/text-data.jsonl.zst', '*/*/omissions.jsonl', '*.jsonl', '*.json',
                    'render-omissions/*/omissions.jsonl', '*/omissions.jsonl'],
        'source_class': ('rendered-natlang-native-teacher-turn-text' if a.kind == 'member'
                         else 'shared-natlang-target-training-set'),
        'split': 'per document, inherited from the source rows; held stays held; train documents sharing a source '
                 'group with any held document (this corpus, the 4962 union, the harness bench) are dropped',
        'admission': 'renders of admitted source corpora: ' + ', '.join(a.source_corpus) +
                     '; variants: ' + variants + '; no new example admission',
        'training_admission': True,
        'training_admission_reason': 'owner 2026-10-11: maximize data; renders of training-admitted corpora '
                                     '(plans/neuralese/DATA_INVENTORY_20261011.md); program-header and whole-trajectory '
                                     'renderings are model-facing changes recorded as debt in plans/MODEL_FACING_CHANGES.md',
        'derived_from': list(a.source_corpus),
        'license': 'facts from the source rows (see the source corpora)',
        **body,
        'notes': a.notes,
    }
    if getattr(a, 'supersedes', None):
        entry['supersedes'] = list(a.supersedes)
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
        subprocess.run([sys.executable, str(ROOT / 'scripts/sync_training_corpora.py'), 'publish', '--machine', a.machine,
                        '--id', a.id], check=True)


if __name__ == '__main__':
    main()
