#!/usr/bin/env python3
"""The shared natlang target training set: membership, per-member files and totals, one immutable revision at a time.

Membership is derived from the registry (origin/main), so every agent's registered renders join without hand edits:

- every entry with ``renders`` (the shared layout ``<path>/<rendering>/<tokenizer>/text-data.jsonl[.zst]``),
  ``training_admission: true``, not retired, not superseded (an entry is superseded when another registered entry
  names it in ``supersedes``, or when its source corpus is superseded that way), and not listed in the spec's
  ``exclude`` (with the reason);
- plus the spec's ``external_members``: registered text sets in their own layout (4962 union and its twins, harness
  bench, SWE whole-trajectory twins), each joining when all its registry ids are registered.

The living spec is ``runs/natlang-target-set-20261011/spec.json`` (data, not git): exclusions, external members and
pending items with the reason each is not integrated. A revision is written as ``r<N>.json`` next to it and registered
as ``natlang-target-set-20261011-r<N>`` by target_set_watcher.py together with the union built from it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'training/neuralese_corpora.json'
SET_DIR = ROOT / 'runs' / 'natlang-target-set-20261011'
SPEC = SET_DIR / 'spec.json'
BASE_ID = 'natlang-target-set-20261011'
TOKENIZERS = ('lfm25', 'mellum21')
RENDERINGS = ('full', 'program-header', 'whole-trajectory')


def origin_registry(fetch: bool = True) -> dict:
    if fetch:
        subprocess.run(['git', 'fetch', '-q', 'origin'], cwd=ROOT, check=False)
    shown = subprocess.run(['git', 'show', f'origin/main:{REGISTRY}'], cwd=ROOT, check=True, capture_output=True,
                           text=True).stdout
    return json.loads(shown)


def superseded_ids(corpora: list[dict]) -> set[str]:
    out = set()
    for c in corpora:
        sup = c.get('supersedes')
        out.update([sup] if isinstance(sup, str) else (sup or []))
        if c.get('superseded_by'):
            out.add(c['id'])
    return out


def member_file(entry_path: str, rendering: str, tok: str) -> str | None:
    for name in ('text-data.jsonl', 'text-data.jsonl.zst'):
        rel = f'{entry_path}/{rendering}/{tok}/{name}'
        if (ROOT / rel).exists():
            return rel
    return None


def resolve(registry: dict, spec: dict) -> tuple[list[dict], list[dict]]:
    """(members, skipped) for the spec over the registry."""
    corpora = registry['corpora']
    by_id = {c['id']: c for c in corpora}
    superseded = superseded_ids(corpora)
    exclude = {item['id']: item['reason'] for item in spec.get('exclude', [])}
    members, skipped = [], []
    for c in corpora:
        if not isinstance(c.get('renders'), dict) or c['id'].startswith(BASE_ID) or c.get('kind') == 'union':
            continue
        if c.get('source_class') == 'shared-natlang-target-training-set':
            continue
        reason = ('not training-admitted' if c.get('training_admission') is not True else
                  'retired' if c.get('retired') else
                  'superseded' if c['id'] in superseded or any(d in superseded for d in c.get('derived_from') or []) else
                  exclude.get(c['id']))
        if reason:
            skipped.append({'id': c['id'], 'reason': reason})
            continue
        files = {}
        for rendering in RENDERINGS:
            per_tok = {tok: member_file(c['path'], rendering, tok) for tok in TOKENIZERS}
            if all(per_tok.values()):
                files[rendering] = per_tok
        if not files:
            skipped.append({'id': c['id'], 'reason': 'no rendering with files for both tokenizers on this machine'})
            continue
        members.append({'id': c['id'], 'derived_from': c.get('derived_from'), 'files': files,
                        'stats': {r: {t: (c['renders'].get(r, {}).get(t) or {}).get('by_split') for t in TOKENIZERS}
                                  for r in files}})
    for ext in spec.get('external_members', []):
        missing = [i for i in ext['registry_ids'] if i not in by_id]
        if missing:
            skipped.append({'id': ext['id'], 'reason': f'not registered yet: {missing}'})
            continue
        if not all((ROOT / p).exists() for toks in ext['files'].values() for p in toks.values()):
            skipped.append({'id': ext['id'], 'reason': 'files missing on this machine'})
            continue
        members.append({'id': ext['id'], 'registry_ids': ext['registry_ids'], 'note': ext.get('note', ''),
                        'files': ext['files']})
    return members, skipped


def signature(members: list[dict]) -> str:
    import hashlib
    key = [(m['id'], sorted((r, t, p) for r, toks in m['files'].items() for t, p in toks.items())) for m in members]
    return hashlib.sha256(json.dumps(sorted(key)).encode()).hexdigest()


def main():
    spec = json.loads(SPEC.read_text())
    members, skipped = resolve(origin_registry(), spec)
    print(json.dumps({'members': [m['id'] for m in members], 'skipped': skipped, 'signature': signature(members)},
                     indent=1))


if __name__ == '__main__':
    main()
