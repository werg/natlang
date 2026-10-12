#!/usr/bin/env python3
"""Measure what a model actually needs on the in-run held task subset: per-task wall time, model requests, turns and
outcome, against an external OpenAI-compatible server (e.g. llama-server), with budgets set far above any binding
value. Used to set task_eval's case_seconds / max_model_requests from measurement (AGENTS.md "No premature limits").

  python3 scripts/held_task_budget_probe.py --spec-from training/neuralese/recipes/neuralese-direct-v2.json \
      --server http://127.0.0.1:8097 --out <data_nvme>/held-budget-probe/run-a --case-seconds 7200 \
      --max-model-requests 1000 --max-turns 1000 --workers 8

Same subset, queue entries, collector and exact scorer as the in-run evaluator (eval/task_cadence.py); only the
server is external and the budgets come from the command line. Writes budget-probe.json (per task: id, family,
outcome, seconds, model requests, turns, resource_limit_reason) and prints the distribution.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.eval import task_cadence  # noqa: E402
from natlang_neuralese.eval.task_packet import read_jsonl, score_rows  # noqa: E402


def recipe_spec(path: Path) -> dict:
    recipe = json.loads(path.read_text())
    found = []

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get('task_eval'), dict) and 'packet' in node['task_eval']:
                found.append(node['task_eval'])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(recipe)
    if not found:
        raise SystemExit(f'no task_eval with a packet in {path}')
    return dict(found[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec-from', type=Path, help='recipe whose task_eval (packet, cases) is measured')
    parser.add_argument('--spec', help='task_eval JSON instead of --spec-from')
    parser.add_argument('--server', required=True, help='OpenAI-compatible base URL')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case-seconds', type=int, default=7200)
    parser.add_argument('--max-model-requests', type=int, default=1000)
    parser.add_argument('--max-turns', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model-concurrency', type=int, default=2)
    args = parser.parse_args(argv)

    spec = json.loads(args.spec) if args.spec else recipe_spec(args.spec_from)
    spec.update(case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                max_turns=args.max_turns, workers=args.workers)
    spec = task_cadence.parse_task_eval(spec)
    evaluation = task_cadence.TaskEval(spec, args.out, None, prompt_rendering='full')
    run = evaluation.root / 'probe'
    run.mkdir(parents=True, exist_ok=True)
    queues = evaluation._queues(run)
    env = task_cadence.node_path_env()
    started = time.time()
    children = []
    for queue, journal, _ in queues:
        log = (queue.parent / 'supervisor.log').open('a')
        children.append((subprocess.Popen(
            [sys.executable, str(ROOT / 'scripts' / 'run_bonsai_queue.py'), str(queue), str(journal),
             '--runtime', str(spec.get('runtime') or ROOT / 'ts-host'), '--server', args.server,
             '--chat-completions-url', args.server.rstrip('/') + '/v1/chat/completions',
             '--model-concurrency', str(args.model_concurrency), '--case-seconds', str(args.case_seconds)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = []
    for child, log in children:
        codes.append(child.wait())
        log.close()

    finishes, results, keys = {}, [], {}
    for queue, journal, entries in queues:
        if journal.is_file():
            for line in journal.read_text().splitlines():
                if line.strip():
                    event = json.loads(line)
                    if event.get('event') == 'finish':
                        finishes[f'{queue.parent.name}/{event["key"]}'] = event
        for entry in entries:
            keys[entry['index']] = f'{queue.parent.name}/{entry["key"]}'
            if Path(entry['output']).is_file():
                results.extend(read_jsonl(entry['output']))
    score = score_rows(evaluation.rows, results, finishes, keys)
    tasks = []
    for index, case in enumerate(score['cases']):
        finish = finishes.get(keys[index], {})
        tasks.append({'case_index': spec['cases'][index], 'id': case['id'], 'family': case['family'],
                      'outcome': case['outcome'], 'seconds': finish.get('elapsed_seconds'),
                      'model_requests': finish.get('fresh_model_replies'), 'turns': finish.get('saved_turns'),
                      'completion_tokens': finish.get('completion_tokens'),
                      'resource_limit_reason': finish.get('resource_limit_reason'),
                      'status': finish.get('status')})
    report = {'schema': 'natlang.held-task-budget-probe/1', 'server': args.server,
              'packet_sha256': spec['packet']['sha256'], 'case_indices': spec['cases'],
              'budgets': {k: spec[k] for k in ('case_seconds', 'max_model_requests', 'max_turns', 'workers')},
              'model_concurrency_per_worker': args.model_concurrency, 'wall_seconds': round(time.time() - started, 1),
              'task_successes': score['task_successes'], 'outcome_counts': score['outcome_counts'],
              'collector_exit_codes': codes, 'tasks': tasks}
    (args.out / 'budget-probe.json').write_text(json.dumps(report, indent=1) + '\n')
    for task in tasks:
        print(f"{task['case_index']:>3} {task['outcome']:<24} {task['seconds']!s:>8}s {task['model_requests']!s:>5} req "
              f"{task['turns']!s:>5} turns {task['resource_limit_reason'] or ''} {task['family'] or ''}")
    print(json.dumps({k: report[k] for k in ('task_successes', 'outcome_counts', 'wall_seconds')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
