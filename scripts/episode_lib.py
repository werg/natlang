"""Shared pieces of skill-episode builders (natlang.skill-episode/1).

Builders differ in their sources; how an episode is assembled should not. This module holds the parts every
builder needs to get the same way: stable digests, `.nl` targets, case records with source groups, the held-out
schedule, group commitments, answer-position balancing for multiple choice, and packet writing with its manifest.
Every packet written here should pass the gate, `ts-host/scripts/skills/audit-episodes.mjs`, before collection.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

GATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ts-host', 'scripts', 'skills', 'audit-episodes.mjs')


def digest(value) -> str:
    return hashlib.sha256((value if isinstance(value, str) else json.dumps(value, sort_keys=True)).encode()).hexdigest()


def nl_target(name: str, args: dict, returns: str, prompt: str, entry: str = 'solve.nl') -> dict:
    """An improvement-case target with one `.nl` function; types are JSON-quoted, which the loader unquotes."""
    return {'kind': 'improvement-case', 'entry': entry, 'exportName': 'default',
            'source': {'schema': 'natlang.skill-graded-target/1', 'id': name},
            'files': {entry: '---\nargs: { ' + ', '.join(f'{arg}: {json.dumps(kind)}' for arg, kind in args.items())
                      + ' }\nreturns: ' + json.dumps(returns) + f'\n---\n{prompt}\n'}}


def case_record(source: str, key: str, args: list, expected) -> dict:
    """A case whose ID and source group derive from its source key: the same item is the same group everywhere."""
    return {'id': 'case-' + digest(f'{source}:{key}')[:20], 'group': 'g-' + digest(f'{source}-group:{key}')[:24],
            'args': args, 'expected': expected}


def held_split(n: int, per_family: int) -> str:
    """The last fifth (at least two) of a family's episodes is held out, alternating validation and test."""
    held = max(2, per_family // 5)
    rank = n - (per_family - held)
    return ('validation' if rank % 2 == 0 else 'test') if rank >= 0 else 'train'


def group_commitment(groups) -> str:
    return 'group-commitment:sha256:' + digest(sorted(groups))


def balance_choice_positions(cases: list[dict], salt: str, options_arg: int = 1) -> list[dict]:
    """Move each multiple-choice case's answer to a rotating label (`args[options_arg]` maps labels to option texts,
    `expected.answer` names the right label), so no answer letter dominates a set of cases and a constant letter
    cannot score. Option texts are permuted; the labels stay in order."""
    out = []
    start = int(digest(salt), 16)
    for index, case in enumerate(cases):
        options = case['args'][options_arg]
        labels = sorted(options)
        answer = case['expected']['answer']
        target = labels[(start + index) % len(labels)]
        texts = [options[label] for label in labels if label != answer]
        order = [label for label in labels if label != target]
        moved = {target: options[answer], **dict(zip(order, texts))}
        args = list(case['args'])
        args[options_arg] = {label: moved[label] for label in labels}
        out.append({**case, 'args': args, 'expected': {**case['expected'], 'answer': target}})
    return out


def write_packet(out_dir: str, stem: str, episodes: list[dict], manifest: dict) -> dict:
    """Write `<stem>.jsonl` (sorted keys, one episode per line) and `<stem>.manifest.json` with its sha256, count and
    splits. Both files are created, never overwritten. Support and query groups must not repeat across episodes."""
    seen = [c['group'] for e in episodes for part in ('support', 'query') for c in e[part]['cases']]
    if len(seen) != len(set(seen)):
        raise ValueError('support/query groups repeat across episodes')
    os.makedirs(out_dir, exist_ok=True)
    body = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in episodes)
    with open(os.path.join(out_dir, f'{stem}.jsonl'), 'x') as stream:
        stream.write(body)
    # Keys keep the caller's order; `episodes`, `splits` and `sha256` fill placeholders there or are appended.
    full = dict(manifest)
    full['episodes'] = len(episodes)
    full['splits'] = {s: sum(1 for e in episodes if e['split'] == s) for s in ['train', 'validation', 'test']}
    full['sha256'] = digest(body)
    with open(os.path.join(out_dir, f'{stem}.manifest.json'), 'x') as stream:
        stream.write(json.dumps(full, indent=2) + '\n')
    return full


def run_gate(paths: list[str], database_root: str | None = None, report: str | None = None) -> dict:
    """Run the episode gate on written packets; raise with its errors when it fails. Warnings are returned."""
    command = ['node', GATE, *paths] + (['--database-root', database_root] if database_root else []) + \
        (['--out', report] if report else [])
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f'episode gate did not run: {result.stderr[-2000:]}')
    if result.returncode:
        raise RuntimeError(f"episode gate failed with {summary['error_count']} errors: {json.dumps(summary['errors'])[:2000]}")
    return summary
