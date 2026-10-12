#!/usr/bin/env python3
"""Generate a machine's short current-state page from live sources (owner decision 2026-10-12, review rec. 9).

The page replaces the hand-written handover logs: plans/neuralese/HANDOVER.md (DGX) and plans/POP_CURRENT.md
(Pop). It is rebuilt, never edited; history lives in git. Sources, in order of authority:

1. the coordination status pages (`scripts/coord.py status`, both machines);
2. running units on this machine (the memory ledger's claims plus `natlang-*` systemd user units), with the
   recipe each one runs;
3. the declared lineages in training/neuralese/line-parity.json and the recipes on disk;
4. the corpus registry training/neuralese_corpora.json (counts only);
5. open coordination requests to the machine and the latest decisions.

Usage:
    python3 scripts/current_state.py --machine dgx            # writes plans/neuralese/HANDOVER.md
    python3 scripts/current_state.py --machine pop            # writes plans/POP_CURRENT.md (run on Pop)
    python3 scripts/current_state.py --machine pop --stdout   # print instead of writing
    ssh pop-os 'NATLANG_REPO=~/natlang python3 - --machine pop --stdout' < scripts/current_state.py   # Pop's page from DGX
Each machine generates its own page (units are read locally). Regenerate when the status page changes, and commit.
"""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(os.environ.get('NATLANG_REPO') or Path(__file__).resolve().parents[1])
PAGES = {'dgx': 'plans/neuralese/HANDOVER.md', 'pop': 'plans/POP_CURRENT.md'}
POINTERS = [
    ('AGENTS.md', 'owner rules (start here)'),
    ('plans/REVIEW_20261012.md', 'strategic review and the accepted direction'),
    ('plans/neuralese/NEURALESE_DIRECT_V1.md', 'the active training design'),
    ('plans/neuralese/LINE_PARITY.md', 'Mellum/LFM line parity'),
    ('plans/neuralese/DECISIONS.md', 'recorded decisions'),
    ('plans/MACHINE_COORDINATION.md', 'coordination protocol, data sync, storage'),
    ('plans/OWNER_REVIEW.md', 'owner review queue'),
    ('plans/MODEL_FACING_CHANGES.md', 'model-facing changes and measurement debt'),
    ('plans/NATLANG_PILLAR.md', 'weekly NatLang-pillar slot'),
]


def run(*command, timeout=60):
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return result.stdout if result.returncode == 0 else ''


def coord(*args):
    return run(sys.executable, 'scripts/coord.py', '--as', 'current-state', *args)


def status_pages():
    pages = {}
    for path in sorted((REPO / '.coordination/status').glob('*.json')):
        try:
            page = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        pages[page['machine']] = page
    return pages


def units():
    """Running work on this machine: ledger claims and natlang-* systemd user units."""
    found = {}
    try:
        ledger = json.loads(run(sys.executable, 'scripts/memory_ledger.py', 'status') or '{}')
    except ValueError:
        ledger = {}
    for name, claim in (ledger.get('claims') or {}).items():
        found[name.removesuffix('.service')] = {
            'priority': claim.get('priority'), 'budget_gb': claim.get('budget_gb'), 'used_gb': claim.get('used_gb'),
            'command': claim.get('command') or ''}
    listing = run('systemctl', '--user', 'list-units', '--type=service', '--state=running', '--no-legend',
                  '--plain', 'natlang-*')
    for line in listing.splitlines():
        parts = line.split(None, 4)
        if not parts:
            continue
        name = parts[0].removesuffix('.service')
        command = parts[4] if len(parts) > 4 else ''
        found.setdefault(name, {'command': command})
        if not found[name].get('command'):
            found[name]['command'] = command
    for unit in found.values():
        match = re.search(r'--recipe[= ](\S+)', unit['command'])
        unit['recipe'] = Path(match.group(1)).name if match else None
    return found


def first_sentence(text, limit=220):
    text = ' '.join((text or '').split())
    end = text.find('. ')
    text = text[:end + 1] if 0 < end < limit else text
    return text if len(text) <= limit else text[:limit - 1] + '…'


def lineages():
    parity = json.loads((REPO / 'training/neuralese/line-parity.json').read_text())
    recipes = REPO / 'training/neuralese/recipes'
    rows = []
    for stage, entry in parity.get('stages', {}).items():
        for line, spec in sorted((entry.get('lines') or {}).items()):
            name = spec.get('recipe')
            chain, description = [], ''
            current = name
            while current and (recipes / f'{current}.json').exists() and current not in chain:
                chain.append(current)
                data = json.loads((recipes / f'{current}.json').read_text())
                description = description or data.get('description', '')
                current = data.get('extends')
            rows.append((stage, line, name, ' → '.join(chain[1:]), first_sentence(description)))
    return rows


def registry():
    corpora = json.loads((REPO / 'training/neuralese_corpora.json').read_text())['corpora']
    live = [c for c in corpora if not c.get('retired')]
    owners = collections.Counter(c.get('owner') for c in live)
    admitted = sum(1 for c in live if c.get('training_admission') is True)
    return len(live), admitted, owners


def decisions(limit=6):
    try:
        messages = json.loads(coord('log', '-n', '400', '--json') or '[]')
    except ValueError:
        return []
    return [m for m in messages if m.get('kind') == 'decision'][-limit:]


def open_requests(machine):
    lines = [line for line in coord('open').splitlines() if line.strip()]
    requests = []
    for head, route in zip(lines[0::2], lines[1::2]):
        if f'→ {machine}' in route or '→ all' in route or re.search(rf'→ [^:]*\b{machine}-', route):
            requests.append((head.split()[1], route.split(':', 1)[-1].strip()))
    return requests


def render(machine, now):
    pages = status_pages()
    other = 'pop' if machine == 'dgx' else 'dgx'
    out = [f'# {machine.upper()} current state',
           '',
           f'Generated {now} by `python3 scripts/current_state.py --machine {machine}`. Do not edit by hand: regenerate.',
           'This page holds the present only. History is in git (`git log -p -- ' + PAGES[machine] + '`), in the',
           'decision records and in the coordination log. Live processes and the status pages are the authority.',
           '']
    out += ['## Read first', ''] + [f'- `{path}`: {why}' for path, why in POINTERS if (REPO / path).exists()] + ['']
    for name in (machine, other):
        page = pages.get(name)
        out += [f'## {name.upper()} status page' + (' (this machine)' if name == machine else ''), '']
        out += ([f"Updated {page['updated_at']} by {page['by']}.", '', page['body'].strip()] if page
                else ['(no status page; run `coord.py status --set`)'])
        out += ['']
    out += ['## Running on this machine', '']
    running = units()
    if running:
        out += ['| Unit | Priority | Budget / used GB | Recipe |', '| --- | --- | --- | --- |']
        for name, unit in sorted(running.items()):
            budget = (f"{unit['budget_gb']} / {unit['used_gb']}" if unit.get('budget_gb') is not None else '')
            out.append(f"| `{name}` | {unit.get('priority') or ''} | {budget} | {unit['recipe'] or ''} |")
    else:
        out.append('Nothing running.')
    out += ['', '## Declared lineages (training/neuralese/line-parity.json: next launch per line)', '',
            '| Stage | Line | Recipe | Extends | Description |', '| --- | --- | --- | --- | --- |']
    for stage, line, name, chain, description in lineages():
        out.append(f'| {stage} | {line} | `{name}` | {chain} | {description} |')
    recipes = sorted(p.stem for p in (REPO / 'training/neuralese/recipes').glob('*.json'))
    out += ['', f'Recipes on disk ({len(recipes)}): ' + ', '.join(f'`{r}`' for r in recipes), '']
    total, admitted, owners = registry()
    out += ['## Corpus registry', '',
            f'{total} live entries in `training/neuralese_corpora.json` ('
            + ', '.join(f'{n} {o}' for o, n in sorted(owners.items())) + f'); {admitted} carry training admission.', '']
    requests = open_requests(machine)
    out += [f'## Open coordination requests to {machine}', '']
    out += [f'- `{rid}`: {subject}' for rid, subject in requests] or ['None.']
    out += ['', '## Latest decisions (coord)', '']
    out += [f"- {m['sent_at'][:16]} {m['from']}: {m.get('subject') or ''}" for m in decisions()] or ['None.']
    return '\n'.join(out) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--machine', choices=sorted(PAGES), default=os.environ.get('COORD_MACHINE'))
    parser.add_argument('--stdout', action='store_true')
    args = parser.parse_args()
    if not args.machine:
        parser.error('--machine is required')
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    page = render(args.machine, now)
    if args.stdout:
        sys.stdout.write(page)
        return
    target = REPO / PAGES[args.machine]
    temporary = target.with_suffix('.tmp')
    temporary.write_text(page)
    os.replace(temporary, target)
    print(target)


if __name__ == '__main__':
    main()
