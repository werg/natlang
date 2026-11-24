#!/usr/bin/env python3
"""The atlas of existing program ideas, and the nearest existing ideas to a new one (plans/NOVEL_PROGRAMS_20261124.md).

The program inventor (`applications/program-inventor/`) shows the generating model this atlas and requires a
completely new idea. The atlas is one line per existing idea, grouped by where it lives:

- every family of the registry role `grpo-pool` (train rows only; one line per family: its name, row count, shape and
  the first sentence of one root instruction);
- every application under `applications/` (its `natlang.json` description), every design brief under
  `codebases/design_briefs/` (title and first stage), every example under `examples/`;
- every dataset source in `scripts/rl_tasks/SOURCES`;
- every idea the inventor has already accepted (`--accepted LEDGER.jsonl`, the inventor's own ledger), so its output
  never repeats itself.

It is rebuilt from those sources at the start of every invention batch, so it is current by construction. Lines are
capped at 220 characters, which keeps ~800 ideas near 45k tokens.

  program_atlas.py build [--accepted LEDGER.jsonl] -o atlas.json [--markdown atlas.md]
  program_atlas.py near atlas.json IDEA.json [-k 12]     nearest atlas entries to an idea card (TF-IDF cosine)

`near` is a crisp retrieval floor for the novelty judgment, not the judgment: it picks which existing ideas the judge
compares the new idea with in full (the k most similar, plus every entry the inventor named as nearest), and records
their similarity.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
LINE = 220
WORD = re.compile(r'[a-z][a-z0-9]+')
STOP = frozenset('the a an of and or to in on for with by from is are be as at that this it its each every one '
                 'which what when then than into their there these those not no all any per over under who whose'.split())


def first_sentence(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    match = re.match(r'(.+?[.!?])(\s|$)', text)
    return (match.group(1) if match else text)


def nl_body(source: str) -> tuple[dict, str]:
    if not source.startswith('---'):
        return {}, source
    _, head, body = source.split('---', 2)
    description = re.search(r'^description:\s*(.+)$', head, re.M)
    return ({'description': description.group(1).strip().strip('"')} if description else {}), body


def clip(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) <= LINE else text[:LINE - 1].rstrip() + '…'


def pool_entries() -> list[dict]:
    from natlang_neuralese.common.data_registry import role_paths
    families: dict[str, dict] = {}
    for path in role_paths('grpo-pool'):
        for line in open(path, encoding='utf-8'):
            row = json.loads(line)
            if row.get('split', 'train') != 'train':
                continue
            family = row.get('family') or '?'
            entry = families.setdefault(family, {'rows': 0, 'shapes': collections.Counter(), 'line': '', 'source': row.get('source', '')})
            entry['rows'] += 1
            shape = (row.get('curriculum') or {}).get('shape')
            if shape:
                entry['shapes'][shape] += 1
            if not entry['line']:
                semantics = row.get('semantics') or {}
                files, root = semantics.get('files') or {}, semantics.get('root')
                if root in files:
                    head, body = nl_body(files[root])
                    entry['line'] = head.get('description') or first_sentence(body)
    out = []
    for family, entry in sorted(families.items()):
        shapes = ', '.join(shape for shape, _ in entry['shapes'].most_common(3))
        kind = 'authored-family' if family.startswith('auth_') else 'pool-family'
        out.append({'id': f'{kind}:{family}', 'kind': kind, 'name': family, 'rows': entry['rows'],
                    'line': clip(f'{family} ({entry["rows"]} rows; {shapes}): {entry["line"]}')})
    return out


def repo_entries() -> list[dict]:
    out = []
    for manifest in sorted((REPO / 'applications').glob('*/natlang.json')):
        description = json.loads(manifest.read_text()).get('description', '')
        out.append({'id': f'application:{manifest.parent.name}', 'kind': 'application', 'name': manifest.parent.name,
                    'line': clip(f'{manifest.parent.name}: {description}')})
    for brief in sorted((REPO / 'codebases' / 'design_briefs').glob('*/brief.json')):
        data = json.loads(brief.read_text())
        stages = data.get('stages') or []
        first = stages[0].get('algorithm', '') if stages and isinstance(stages[0], dict) else ''
        episode = data.get('representative_episode', '')
        out.append({'id': f'design-brief:{brief.parent.name}', 'kind': 'design-brief', 'name': data.get('title', brief.parent.name),
                    'line': clip(f'{data.get("title", brief.parent.name)}: {episode or first_sentence(first)}')})
    for example in sorted(p for p in (REPO / 'examples').iterdir() if p.is_dir()):
        manifest = example / 'natlang.json'
        description = json.loads(manifest.read_text()).get('description', '') if manifest.exists() else ''
        out.append({'id': f'example:{example.name}', 'kind': 'example', 'name': example.name,
                    'line': clip(f'{example.name}: {description}')})
    for line in (REPO / 'scripts' / 'rl_tasks' / 'SOURCES').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, target = line.split()[:2]
            out.append({'id': f'dataset:{name}', 'kind': 'dataset', 'name': name,
                        'line': clip(f'{name} (dataset source, {target})')})
    return out


def accepted_entries(ledger: Path | None) -> list[dict]:
    if not ledger or not ledger.exists():
        return []
    out = []
    for line in ledger.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            idea = record['idea']
            out.append({'id': f'invented:{idea["slug"]}', 'kind': 'invented', 'name': idea['slug'],
                        'line': clip(f'{idea["slug"]}: {idea["pitch"]} Judgments: {"; ".join(idea.get("judgments", []))}')})
    return out


GROUPS = (('invented', 'Ideas this process already accepted (never repeat or vary these)'),
          ('application', 'Applications (applications/)'), ('example', 'Examples (examples/)'),
          ('design-brief', 'Design-brief programs (codebases/design_briefs/)'),
          ('authored-family', 'Authored task families in training (grpo-pool, auth_*)'),
          ('pool-family', 'Other task families in training (grpo-pool: dataset-derived, composed, codebase)'),
          ('dataset', 'Dataset sources (scripts/rl_tasks/SOURCES)'))


def markdown(entries: list[dict]) -> str:
    by_kind = collections.defaultdict(list)
    for entry in entries:
        by_kind[entry['kind']].append(entry)
    parts = [f'# Atlas of existing program ideas ({len(entries)} entries)\n']
    for kind, title in GROUPS:
        if by_kind[kind]:
            parts.append(f'\n## {title} ({len(by_kind[kind])})\n\n' + ''.join(f'- {e["line"]}\n' for e in by_kind[kind]))
    return ''.join(parts)


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower().replace('_', ' ').replace('-', ' ')) if w not in STOP]


def idea_text(idea: dict) -> str:
    return ' '.join(str(idea.get(key, '')) for key in ('slug', 'pitch', 'user', 'judgments', 'shape', 'inputs', 'output'))


def near(entries: list[dict], idea: dict, k: int) -> list[dict]:
    docs = [tokens(e['line']) for e in entries]
    df = collections.Counter(w for doc in docs for w in set(doc))
    n = len(docs)

    def vector(words):
        counts = collections.Counter(words)
        vec = {w: (1 + math.log(c)) * math.log((n + 1) / (df.get(w, 0) + 1)) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    query = vector(tokens(idea_text(idea)))
    scored = []
    for entry, doc in zip(entries, docs):
        vec = vector(doc)
        scored.append((sum(query.get(w, 0) * v for w, v in vec.items()), entry))
    scored.sort(key=lambda item: -item[0])
    out = [{**entry, 'similarity': round(score, 4)} for score, entry in scored[:k]]
    # The entries the inventor itself named as nearest (by the name its line starts with, or by id) join the
    # retrieved ones, so the judge reads both what word overlap finds and what the inventor compared itself with.
    named = set(idea.get('nearest') or [])
    for score, entry in scored[k:]:
        if entry['id'] in named or entry.get('name') in named or entry['line'].split(':', 1)[0] in named:
            out.append({**entry, 'similarity': round(score, 4)})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--accepted', type=Path)
    b.add_argument('-o', '--out', type=Path, required=True)
    b.add_argument('--markdown', type=Path)
    q = sub.add_parser('near')
    q.add_argument('atlas', type=Path)
    q.add_argument('idea', type=Path)
    q.add_argument('-k', type=int, default=12)
    args = parser.parse_args(argv)
    if args.command == 'build':
        commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        entries = accepted_entries(args.accepted) + repo_entries() + pool_entries()
        args.out.write_text(json.dumps({'schema': 'natlang.program-atlas/1', 'commit': commit, 'entries': entries},
                                       ensure_ascii=False, indent=1) + '\n')
        text = markdown(entries)
        if args.markdown:
            args.markdown.write_text(text)
        print(json.dumps({'entries': len(entries), 'chars': len(text), 'approx_tokens': len(text) // 4,
                          'by_kind': collections.Counter(e['kind'] for e in entries)}))
    else:
        atlas = json.loads(args.atlas.read_text())
        idea = json.loads(args.idea.read_text())
        print(json.dumps(near(atlas['entries'], idea.get('idea', idea), args.k), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
