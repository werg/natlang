#!/usr/bin/env python3
"""Paired live A/B of model-facing runtime variants on a served policy (AGENTS.md "Measure model-facing changes live").

Reuses the RL task-rollout pipeline: queue entries from `rl.task_rollouts.build_queue`, the production collector
(`scripts/run_bonsai_queue.py`, ts-host dist/teacher/cli.js) against an already-served OpenAI-compatible policy, and the
production classifier (`rl.task_rollouts.score`). Arms differ only in the `NATLANG_MODEL_FACING` environment variable of
the collector (`ts-host/src/runtime/model-facing.ts`); every arm gets the same tasks, seeds and sample order (sample-major,
so a partly finished run stays balanced), and the arms run concurrently so server load is shared.

  prepare  --packet P --rows 3,14,15 --samples 8 --arms baseline=none,combined=combined --out DIR
  run      --out DIR --server http://127.0.0.1:8200 [--runtime RT] [--case-seconds N]
  report   --out DIR

Training-split packet rows only (prepare refuses any other split). The server is never started here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.rl import task_rollouts  # noqa: E402
from natlang_neuralese.eval.task_packet import read_jsonl  # noqa: E402


def prepare(args) -> None:
    out = args.out.resolve()
    rows = [read_jsonl(args.packet)[int(index)] for index in args.rows.split(',')]
    if any(row.get('split') != 'train' for row in rows):
        raise SystemExit('only training-split rows may be measured')
    arms = dict(item.split('=', 1) for item in args.arms.split(','))
    for arm, variant in arms.items():
        directory = out / arm
        directory.mkdir(parents=True, exist_ok=True)
        packet = directory / 'packet.jsonl'
        packet.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        entries = task_rollouts.build_queue(packet, len(rows), args.samples, directory, temperature=args.temperature,
                                            case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                                            max_turns=args.max_turns, seed_base=args.seed_base)
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=32768)
        entries.sort(key=lambda entry: (entry['sample'], entry['index']))
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in entries))
        (directory / 'preparation.json').write_text(json.dumps({
            'schema': 'natlang.task-rollout-preparation/1', 'root_mode': 'student', 'model_facing': variant,
            'source_packet': str(args.packet), 'source_rows': args.rows,
            'packet_sha256': hashlib.sha256(packet.read_bytes()).hexdigest()}, indent=2) + '\n')
    (out / 'arms.json').write_text(json.dumps(arms, indent=2) + '\n')
    print(json.dumps({'arms': arms, 'tasks': [row['id'] for row in rows], 'entries_per_arm': len(rows) * args.samples}))


def run(args) -> None:
    out = args.out.resolve()
    arms = json.loads((out / 'arms.json').read_text())
    config = out / 'sampling-config.json'
    config.write_text(json.dumps({'guidance': False}) + '\n')
    children = []
    for arm, variant in arms.items():
        env = dict(os.environ, NATLANG_MODEL_FACING=variant)
        log = (out / arm / 'queue.log').open('a')
        command = [sys.executable, str(ROOT / 'scripts' / 'run_bonsai_queue.py'), str(out / arm / 'queue.jsonl'),
                   str(out / arm / 'journal.jsonl'), '--runtime', str(args.runtime), '--server', args.server,
                   '--chat-completions-url', args.server.rstrip('/') + '/v1/chat/completions', '--model-id', args.model_id,
                   '--model-concurrency', '1', '--chat-request-config', str(config), '--neuralese-server']
        children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    print(json.dumps({'pids': [child.pid for child in children]}), flush=True)
    for child in children:
        child.wait()


STUB_WORDS = re.compile(r'Decide whether item meets policy|Refunds are allowed within 30 days')
ZERO_ARG_CHILD = re.compile(r'\bnl(?:\.with)?(?:<[^`(]*>)?(?:\([^`]*\))?`[^`]*`\(\s*\)')


def behaviours(directory: Path) -> dict:
    """What each change targets, counted over the arm's actual actions (each action and tool result counted once)."""
    counts = dict(episodes=0, turns=0, evals=0, repeated_eval_code=0, zero_arg_children=0, zero_arg_refusals=0,
                  prompt_example_copies=0, receipts=0, footers=0, read_file_calls=0, example_in_system_prompt=0)
    for results in sorted(directory.glob('row-*/sample-*/results.jsonl')):
        for row in read_jsonl(results):
            counts['episodes'] += 1
            codes = {}
            for turn in row.get('trajectory') or []:
                if turn.get('phase') != 'action':
                    continue
                counts['turns'] += 1
                context = turn.get('context') or []
                if context and context[0].get('role') == 'system' and 'Is draft clear to audience' in str(context[0].get('content')):
                    counts['example_in_system_prompt'] += 1
                last = context[-1] if context else {}
                if last.get('role') == 'tool':  # the result of the previous action, seen once per turn
                    result = str(last.get('content'))
                    counts['zero_arg_refusals'] += 'was given no arguments' in result
                    counts['receipts'] += 'Service calls completed by this eval' in result
                    counts['footers'] += '(in eval: const text = await folder.file(' in result
                for name, args in (turn.get('model_response') or {}).get('calls') or []:
                    counts['read_file_calls'] += name == 'read_file'
                    if name != 'eval':
                        continue
                    code = str((args or {}).get('code', ''))
                    counts['evals'] += 1
                    counts['prompt_example_copies'] += len(STUB_WORDS.findall(code))
                    counts['zero_arg_children'] += len(ZERO_ARG_CHILD.findall(code))
                    key = (turn.get('invocation_id'), code)
                    counts['repeated_eval_code'] += key in codes
                    codes[key] = True
    return counts


def report(args) -> None:
    out = args.out.resolve()
    arms = json.loads((out / 'arms.json').read_text())
    summary = {}
    for arm in arms:
        score = task_rollouts.score(out / arm)
        rewards = [json.loads(line) for line in (out / arm / 'rewards.jsonl').read_text().splitlines()]
        finished = [row for row in rewards if row['finish_status'] == 'complete']
        by_task = {}
        for row in finished:
            by_task.setdefault(row['task_id'][:60], []).append(row['outcome'] == 'task_success')
        summary[arm] = {'variant': arms[arm], 'finished': len(finished), 'successes': sum(row['outcome'] == 'task_success'
                        for row in finished), 'outcomes': score['outcome_counts'],
                        'by_task': {task: f'{sum(v)}/{len(v)}' for task, v in by_task.items()},
                        'behaviours': behaviours(out / arm)}
    summary['paired'] = paired(out, list(arms))
    print(json.dumps(summary, indent=1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


def paired(out: Path, arms: list[str]) -> dict:
    """Matched (task, sample) episodes finished in both arms: successes, discordant pairs, turns and tokens."""
    rows = {}
    for arm in arms:
        rewards = {(r['task_id'], r['sample']): r for r in map(json.loads, (out / arm / 'rewards.jsonl').read_text().splitlines())}
        finishes = {}
        for line in (out / arm / 'journal.jsonl').read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'finish' and event.get('status') == 'complete':  # an incomplete run is infrastructure, not a result
                finishes[event['key']] = event
        keys = {e['key']: (e['task_id'], e['sample']) for e in map(json.loads, (out / arm / 'queue.jsonl').read_text().splitlines())}
        rows[arm] = {keys[k]: (rewards[keys[k]]['outcome'] == 'task_success', f.get('saved_turns'), f.get('completion_tokens'),
                              f.get('elapsed_seconds')) for k, f in finishes.items()}
    first, second = arms[0], arms[1]
    common = sorted(set(rows[first]) & set(rows[second]))
    only_first = sum(rows[first][k][0] and not rows[second][k][0] for k in common)
    only_second = sum(rows[second][k][0] and not rows[first][k][0] for k in common)
    mean = lambda arm, i: round(sum(rows[arm][k][i] for k in common) / max(len(common), 1), 1)
    return {'pairs': len(common), f'{first}_successes': sum(rows[first][k][0] for k in common),
            f'{second}_successes': sum(rows[second][k][0] for k in common),
            f'only_{first}': only_first, f'only_{second}': only_second,
            f'{first}_mean_turns': mean(first, 1), f'{second}_mean_turns': mean(second, 1),
            f'{first}_mean_tokens': mean(first, 2), f'{second}_mean_tokens': mean(second, 2)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--rows', required=True, help='comma-separated packet row indices')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--arms', default='baseline=none,combined=combined')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--case-seconds', type=int, default=1800)
    p.add_argument('--max-model-requests', type=int, default=64)
    p.add_argument('--max-turns', type=int, default=128)
    p.add_argument('--seed-base', type=int, default=4100000)
    r = sub.add_parser('run')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--server', required=True)
    r.add_argument('--runtime', type=Path, default=ROOT / 'ts-host')
    r.add_argument('--model-id', default='natlang-rl')
    q = sub.add_parser('report')
    q.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    {'prepare': prepare, 'run': run, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
