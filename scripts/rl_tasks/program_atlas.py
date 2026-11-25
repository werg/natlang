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

The claim registry lets many inventors work at once (ROOT: data_nvme:natlang-novel-programs, or NATLANG_INVENTOR_ROOT):

  program_atlas.py atlas [--root ROOT] -o atlas.md       the atlas an inventor reads: repository, pool, every accepted
                                                         idea (ROOT/accepted.jsonl) and every idea claimed for building
  program_atlas.py claim IDEA.json --by AGENT [--root ROOT] [--threshold 0.25]
      register an idea card before building it. Under ROOT/claims/.lock: the slug must be free, and no accepted or
      claimed idea card, nor any atlas line, may be a near duplicate (TF-IDF cosine >= threshold, default 0.35: with the
      full atlas's word weights, distinct pilot ideas score 0.16-0.21 card against card, the 75-card bank's nearest
      pairs median 0.14, max 0.25; a card cut to its pitch, two judgments and inputs scores 0.70-0.72 against its own
      full card). Prints
      {ok, slug, conflicts, nearest}; exit 1 when refused. Appends to ROOT/claims/claims.jsonl (append-only).
  program_atlas.py claims [--root ROOT]                  the latest state of every claim
  program_atlas.py release SLUG --status abandoned|accepted|failed [--root ROOT] [--submission DIR] [--reason TEXT]
      a claim's end (`invented.py admit` records accepted itself)

An idea bank (a directory of idea cards, one JSON file each, written centrally) is handed out one card per builder:

  program_atlas.py bank-check BANK [--root ROOT]         every card against the atlas, the accepted and claimed ideas
                                                         and the bank's other cards: near duplicates and taken slugs
  program_atlas.py take BANK --by AGENT [--root ROOT]    the next free card, claimed under the lock (the claim check
      above); a card that is a near duplicate of an accepted or claimed idea is recorded `duplicate` and skipped.
      Prints {ok, slug, card, nearest} or {ok: false} when the bank is exhausted. The builder ends its claim with
      `invented.py admit` (accepted) or `release SLUG --status failed --reason ...`.

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
        # inv_* families are the inventor's own ideas: listed with them (build() drops those its ledgers already list)
        kind = 'authored-family' if family.startswith('auth_') else 'invented' if family.startswith('inv_') else 'pool-family'
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


def judgment_names(idea: dict) -> str:
    return '; '.join(j.get('name') or j.get('question', '') if isinstance(j, dict) else str(j)
                     for j in idea.get('judgments') or [])


def accepted_entries(ledgers) -> list[dict]:
    """Ideas of the inventor's ledgers (each line `{"idea": IdeaCard, ...}`), first occurrence of a slug kept."""
    out, seen = [], set()
    for ledger in [ledgers] if isinstance(ledgers, (str, Path)) else ledgers or []:
        if not ledger or not Path(ledger).exists():
            continue
        for line in Path(ledger).read_text().splitlines():
            if line.strip():
                idea = json.loads(line)['idea']
                if idea['slug'] in seen:
                    continue
                seen.add(idea['slug'])
                out.append({'id': f'invented:{idea["slug"]}', 'kind': 'invented', 'name': idea['slug'],
                            'line': clip(f'{idea["slug"]}: {idea["pitch"]} Judgments: {judgment_names(idea)}'),
                            'text': idea_text(idea)})
    return out


GROUPS = (('invented', 'Ideas this process already accepted or is building (never repeat or vary these)'),
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
    docs = [tokens(e.get('text') or e['line']) for e in entries]
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
    out = [{**{key: v for key, v in entry.items() if key != 'text'}, 'similarity': round(score, 4)}
           for score, entry in scored[:k]]
    # The entries the inventor itself named as nearest (by the name its line starts with, or by id) join the
    # retrieved ones, so the judge reads both what word overlap finds and what the inventor compared itself with.
    named = set(idea.get('nearest') or [])
    for score, entry in scored[k:]:
        if entry['id'] in named or entry.get('name') in named or entry['line'].split(':', 1)[0] in named:
            out.append({**{key: v for key, v in entry.items() if key != 'text'}, 'similarity': round(score, 4)})
    return out


# ---- the claim registry of concurrent inventors -------------------------------------------------------------------
import fcntl  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

ROOT = Path(os.environ.get('NATLANG_INVENTOR_ROOT') or data_path('data_nvme', 'natlang-novel-programs'))
THRESHOLD = 0.35   # refuse at or above
WATCH = 0.22       # bank-check lists pairs at or above for a look
BASE_MAX_AGE = 3600   # seconds before the repository + pool part of the atlas is rebuilt


def claims_dir(root: Path) -> Path:
    path = Path(root) / 'claims'
    path.mkdir(parents=True, exist_ok=True)
    return path


def claim_states(root: Path) -> dict[str, dict]:
    """The latest record of every claimed slug (claims.jsonl is append-only; a later record of a slug wins)."""
    states: dict[str, dict] = {}
    path = claims_dir(root) / 'claims.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                states[record['slug']] = {**states.get(record['slug'], {}), **record}
    return states


def base_entries(root: Path) -> list[dict]:
    """Repository and pool entries, cached in ROOT/claims/atlas-base.json for BASE_MAX_AGE (reading the pool takes
    about a minute); rebuilt atomically when older."""
    path = claims_dir(root) / 'atlas-base.json'
    if path.exists() and time.time() - path.stat().st_mtime < BASE_MAX_AGE:
        return json.loads(path.read_text())['entries']
    entries = repo_entries() + pool_entries()
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps({'entries': entries}, ensure_ascii=False))
    os.replace(tmp, path)
    return entries


def campaign_entries(root: Path) -> list[dict]:
    """The atlas of a campaign: accepted ideas (ROOT/accepted.jsonl), ideas claimed and being built, repository, pool."""
    accepted = accepted_entries([Path(root) / 'accepted.jsonl'])
    known = {e['name'] for e in accepted}
    building = []
    for slug_, state in claim_states(root).items():
        if state.get('status') == 'claimed' and slug_ not in known:
            idea = state['idea']
            building.append({'id': f'invented:{slug_}', 'kind': 'invented', 'name': slug_, 'claimed_by': state.get('by'),
                             'line': clip(f'{slug_} (being built): {idea.get("pitch", "")} Judgments: {judgment_names(idea)}'),
                             'text': idea_text(idea)})
    listed = {'inv_' + re.sub(r'[^a-z0-9]+', '_', e['name']).strip('_') for e in accepted + building}
    return accepted + building + [e for e in base_entries(root) if e['name'] not in listed]


def claim(root: Path, idea: dict, by: str, threshold: float = THRESHOLD) -> dict:
    with open(claims_dir(root) / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = campaign_entries(root)
        slug_ = str(idea.get('slug') or '')
        problems = []
        if not re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)*', slug_):
            problems.append('slug is lowercase words joined by dashes')
        states = claim_states(root)
        if slug_ in states and states[slug_].get('status') not in ('abandoned', 'failed') or \
                any(e['id'] == f'invented:{slug_}' and e.get('claimed_by') is None for e in entries):
            problems.append(f'slug {slug_} is taken')
        # its own failed or abandoned claim is no neighbour of a retry
        nearest = near([e for e in entries if e['id'] != f'invented:{slug_}'], idea, 12)
        conflicts = [e for e in nearest if e['similarity'] >= threshold]
        problems += [f'near duplicate of {e["id"]} (similarity {e["similarity"]})' for e in conflicts]
        result = {'ok': not problems, 'slug': slug_, 'problems': problems, 'conflicts': conflicts, 'nearest': nearest}
        if not problems:
            with open(claims_dir(root) / 'claims.jsonl', 'a') as out:
                out.write(json.dumps({'slug': slug_, 'status': 'claimed', 'by': by, 'at': time.time(), 'idea': idea,
                                      'nearest': [{'id': e['id'], 'similarity': e['similarity']} for e in nearest[:3]]},
                                     ensure_ascii=False) + '\n')
                out.flush()
                os.fsync(out.fileno())
        return result


def release(root: Path, slug_: str, status: str, submission: str | None = None, reason: str | None = None) -> dict:
    if status not in ('abandoned', 'accepted', 'failed', 'duplicate'):
        raise ValueError('status is abandoned, accepted, failed or duplicate')
    with open(claims_dir(root) / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if slug_ not in claim_states(root):
            raise SystemExit(f'{slug_} was never claimed')
        record = {'slug': slug_, 'status': status, 'at': time.time(), **({'submission': submission} if submission else {}),
                  **({'reason': reason} if reason else {})}
        with open(claims_dir(root) / 'claims.jsonl', 'a') as out:
            out.write(json.dumps(record) + '\n')
        return record


def bank_cards(bank: Path) -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(Path(bank).glob('*.json')):
        card = json.loads(path.read_text())
        out.append((path, card.get('idea', card)))
    return out


def bank_check(root: Path, bank: Path, threshold: float = THRESHOLD) -> dict:
    """Each bank card against the campaign atlas and the bank's earlier cards (file order)."""
    base = campaign_entries(root)
    cards = bank_cards(bank)
    report, earlier = [], []
    for path, card in cards:
        slug_ = str(card.get('slug') or '')
        entries = [e for e in base if e['id'] != f'invented:{slug_}' or e.get('claimed_by') is not None] + earlier
        nearest = near([e for e in entries if e['id'] != f'invented:{slug_}' or e.get('kind') != 'bank'], card, 5)
        taken = any(e['id'] == f'invented:{slug_}' and e.get('claimed_by') is None for e in base) or \
            any(e['name'] == slug_ for e in earlier)
        conflicts = [e for e in nearest if e['similarity'] >= threshold]
        report.append({'file': path.name, 'slug': slug_, 'ok': not conflicts and not taken, 'slug_taken': taken,
                       'conflicts': [{'id': e['id'], 'similarity': e['similarity']} for e in conflicts],
                       'nearest': [{'id': e['id'], 'similarity': e['similarity']} for e in nearest[:3]]})
        earlier.append({'id': f'bank:{slug_}', 'kind': 'bank', 'name': slug_, 'line': clip(f'{slug_}: {card.get("pitch", "")}'),
                        'text': idea_text(card)})
    watch = sorted(({'slug': r['slug'], **r['nearest'][0]} for r in report if r['ok'] and r['nearest']
                    and r['nearest'][0]['similarity'] >= WATCH), key=lambda item: -item['similarity'])
    return {'bank': str(bank), 'cards': len(cards), 'ok': sum(r['ok'] for r in report), 'threshold': threshold,
            'refused': [r for r in report if not r['ok']], 'watch': watch, 'cards_checked': report}


def take(root: Path, bank: Path, by: str, threshold: float = THRESHOLD) -> dict:
    """The next free card of the bank, claimed for `by` (one builder per card)."""
    for path, card in bank_cards(bank):
        states = claim_states(root)
        if card.get('slug') in states and states[card['slug']].get('status') not in ('abandoned',):
            continue
        result = claim(root, {**card, 'bank_file': str(path)}, by, threshold)
        if result['ok']:
            return {'ok': True, 'slug': card['slug'], 'file': str(path), 'card': card, 'nearest': result['nearest'][:5]}
        if card.get('slug') and not any('slug' in p and 'taken' in p for p in result['problems']):
            with open(claims_dir(root) / 'claims.jsonl', 'a') as out:
                out.write(json.dumps({'slug': card['slug'], 'status': 'duplicate', 'by': by, 'at': time.time(),
                                      'idea': card, 'reason': '; '.join(result['problems'])}) + '\n')
    return {'ok': False, 'reason': 'no free card left in the bank'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--accepted', type=Path, action='append', default=[],
                   help='an inventor ledger (repeatable): accepted ideas and ideas being built')
    b.add_argument('-o', '--out', type=Path, required=True)
    b.add_argument('--markdown', type=Path)
    at = sub.add_parser('atlas')
    at.add_argument('--root', type=Path, default=ROOT)
    at.add_argument('-o', '--out', type=Path, required=True)
    cl = sub.add_parser('claim')
    cl.add_argument('idea', type=Path)
    cl.add_argument('--by', required=True, help='the claiming agent (e.g. sonnet-17)')
    cl.add_argument('--root', type=Path, default=ROOT)
    cl.add_argument('--threshold', type=float, default=THRESHOLD)
    ls = sub.add_parser('claims')
    ls.add_argument('--root', type=Path, default=ROOT)
    rl = sub.add_parser('release')
    rl.add_argument('slug')
    rl.add_argument('--status', required=True, choices=('abandoned', 'accepted', 'failed'))
    rl.add_argument('--root', type=Path, default=ROOT)
    rl.add_argument('--submission')
    rl.add_argument('--reason')
    bc = sub.add_parser('bank-check')
    bc.add_argument('bank', type=Path)
    bc.add_argument('--root', type=Path, default=ROOT)
    bc.add_argument('--threshold', type=float, default=THRESHOLD)
    tk = sub.add_parser('take')
    tk.add_argument('bank', type=Path)
    tk.add_argument('--by', required=True)
    tk.add_argument('--root', type=Path, default=ROOT)
    tk.add_argument('--threshold', type=float, default=THRESHOLD)
    q = sub.add_parser('near')
    q.add_argument('atlas', type=Path)
    q.add_argument('idea', type=Path)
    q.add_argument('-k', type=int, default=12)
    args = parser.parse_args(argv)
    if args.command == 'build':
        commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        accepted = accepted_entries(args.accepted)
        listed = {'inv_' + re.sub(r'[^a-z0-9]+', '_', e['name']).strip('_') for e in accepted}
        entries = accepted + repo_entries() + [e for e in pool_entries() if e['name'] not in listed]
        args.out.write_text(json.dumps({'schema': 'natlang.program-atlas/1', 'commit': commit, 'entries': entries},
                                       ensure_ascii=False, indent=1) + '\n')
        text = markdown(entries)
        if args.markdown:
            args.markdown.write_text(text)
        print(json.dumps({'entries': len(entries), 'chars': len(text), 'approx_tokens': len(text) // 4,
                          'by_kind': collections.Counter(e['kind'] for e in entries)}))
    elif args.command == 'atlas':
        entries = campaign_entries(args.root)
        args.out.write_text(markdown(entries))
        print(json.dumps({'entries': len(entries), 'by_kind': collections.Counter(e['kind'] for e in entries)}))
    elif args.command == 'claim':
        idea = json.loads(args.idea.read_text())
        result = claim(args.root, idea.get('idea', idea), args.by, args.threshold)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result['ok'] else 1
    elif args.command == 'claims':
        print(json.dumps({s: {k: v for k, v in st.items() if k != 'idea'} for s, st in claim_states(args.root).items()},
                         indent=1))
    elif args.command == 'release':
        print(json.dumps(release(args.root, args.slug, args.status, args.submission, args.reason)))
    elif args.command == 'bank-check':
        report = bank_check(args.root, args.bank, args.threshold)
        print(json.dumps({k: v for k, v in report.items() if k != 'cards_checked'}, ensure_ascii=False, indent=1))
        (claims_dir(args.root) / f'bank-check-{Path(args.bank).parent.name}.json').write_text(json.dumps(report, indent=1))
        return 0 if not report['refused'] else 1
    elif args.command == 'take':
        result = take(args.root, args.bank, args.by, args.threshold)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result['ok'] else 1
    else:
        atlas = json.loads(args.atlas.read_text())
        idea = json.loads(args.idea.read_text())
        print(json.dumps(near(atlas['entries'], idea.get('idea', idea), args.k), ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
