#!/usr/bin/env python3
"""Measure what a model actually needs on the in-run held task subset: per-task wall time, model requests, turns and
outcome, against an external OpenAI-compatible server (e.g. llama-server), with budgets set far above any binding
value. Used to set task_eval's case_seconds / max_model_requests from measurement (AGENTS.md "No premature limits").

  python3 scripts/held_task_budget_probe.py --spec-from training/neuralese/recipes/RECIPE.json \
      --server http://127.0.0.1:8097 --out <data_nvme>/held-budget-probe/run-a --case-seconds 7200 \
      --max-model-requests 1000 --max-turns 1000 --workers 8

Same subset, queue entries, collector and exact scorer as the in-run evaluator (eval/task_cadence.py); only the
server is external and the budgets come from the command line. Writes budget-probe.json (per task: id, family,
outcome, seconds, model requests, turns, resource_limit_reason) and prints the distribution.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.eval import task_cadence  # noqa: E402
from natlang_neuralese.eval.task_packet import read_jsonl, score_rows  # noqa: E402


def collect_queues(queues, spec, args, env):
    """Use the production slot dispatcher and own each collector's process group."""
    if not queues:
        return []
    children = []
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    def signal_group(child, value):
        try:
            os.killpg(child.pid, value)
        except ProcessLookupError:
            pass

    class Collector:
        def __init__(self, child): self.child = child
        def poll(self): return self.child.poll()
        def terminate(self): signal_group(self.child, signal.SIGTERM)

    def launch(queue, journal, entries):
        log = (queue.parent / 'supervisor.log').open('a')
        try:
            child = subprocess.Popen(
                [sys.executable, str(ROOT / 'scripts' / 'run_bonsai_queue.py'), str(queue), str(journal),
                 '--runtime', str(spec.get('runtime') or ROOT / 'ts-host'), '--server', args.server,
                 '--chat-completions-url', args.server.rstrip('/') + '/v1/chat/completions',
                 '--model-id', args.model_id,
                 '--model-concurrency', str(args.model_concurrency), '--case-seconds', str(args.case_seconds)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        children.append((child, log))
        return Collector(child), log

    handlers = {value: signal.signal(value, request_stop) for value in (signal.SIGTERM, signal.SIGINT)}
    try:
        return task_cadence.wait_collectors([], queues, launch, min_free_mib=0,
            max_workers=min(spec['workers'], len(queues)), stop_requested=lambda: stopping)
    finally:
        for child, log in children:
            signal_group(child, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_group(child, signal.SIGKILL)
                child.wait(timeout=5)
            log.close()
        for value, handler in handlers.items():
            signal.signal(value, handler)


def served_model_id(server: str) -> str:
    """The id the server reports for its model, so results do not carry the collector's default teacher name."""
    with urllib.request.urlopen(server.rstrip('/') + '/v1/models', timeout=30) as response:
        return json.loads(response.read())['data'][0]['id']


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


def apply_cli_budgets(spec: dict, args) -> dict:
    """Apply explicit CLI limits without erasing limits already bound in JSON."""
    defaults = {'case_seconds': 7200, 'max_model_requests': 1000, 'max_turns': 1000, 'workers': 8}
    for name in defaults:
        value = getattr(args, name)
        if value is not None:
            spec[name] = value
        else:
            spec.setdefault(name, defaults[name])
    return spec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec-from', type=Path, help='recipe whose task_eval (packet, cases) is measured')
    parser.add_argument('--spec', help='task_eval JSON instead of --spec-from')
    parser.add_argument('--server', required=True, help='OpenAI-compatible base URL')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case-seconds', type=int)
    parser.add_argument('--max-model-requests', type=int)
    parser.add_argument('--max-turns', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--model-concurrency', type=int, default=2)
    parser.add_argument('--cases', help='comma-separated packet row indices instead of the spec subset (e.g. the '
                        'tasks a stopped probe did not finish)')
    parser.add_argument('--report-only', action='store_true', help='rescore an existing --out without running')
    parser.add_argument('--model-id', help='model id recorded in each result\'s provenance and sent to the server '
                        '(default: the first id the server lists at /v1/models)')
    args = parser.parse_args(argv)
    model_id = args.model_id = args.model_id or (None if args.report_only else served_model_id(args.server))

    if args.spec and not args.spec.lstrip().startswith('{'):
        args.spec = Path(args.spec).read_text()  # a task_eval JSON file (lfm3_serve.sh eval passes a path)
    spec = json.loads(args.spec) if args.spec else recipe_spec(args.spec_from)
    if args.cases:
        spec['cases'] = [int(index) for index in args.cases.split(',')]
    # Explicit CLI values take precedence; otherwise retain the task_eval's
    # declared limits, falling back to the historical probe defaults only when
    # the JSON omitted a field.
    spec = apply_cli_budgets(spec, args)
    spec = task_cadence.parse_task_eval(spec)
    evaluation = task_cadence.TaskEval(spec, args.out, None, prompt_rendering='full')
    run = evaluation.root / 'probe'
    run.mkdir(parents=True, exist_ok=True)
    queues = evaluation._queues(run)
    env = task_cadence.node_path_env()
    started = time.time()
    codes = collect_queues([] if args.report_only else queues, spec, args, env)

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
    def request_count(row):
        starts = (row.get('request_telemetry') or {}).get('starts')
        return len(starts) if isinstance(starts, list) else starts
    requests_by_id = {row['task']['program_ir']['id']: request_count(row) for row in results}
    tasks = []
    for index, case in enumerate(score['cases']):
        finish = finishes.get(keys[index], {})
        requests = requests_by_id.get(case['id'])
        tasks.append({'case_index': spec['cases'][index], 'id': case['id'], 'family': case['family'],
                      'outcome': case['outcome'], 'seconds': finish.get('elapsed_seconds'),
                      'model_requests': requests if requests is not None else finish.get('fresh_model_replies'), 'turns': finish.get('saved_turns'),
                      'completion_tokens': finish.get('completion_tokens'),
                      'resource_limit_reason': finish.get('resource_limit_reason'),
                      'status': finish.get('status')})
    report = {'schema': 'natlang.held-task-budget-probe/1', 'server': args.server, 'model_id': model_id,
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
