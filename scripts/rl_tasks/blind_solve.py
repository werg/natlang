#!/usr/bin/env python3
"""Blind-solve packets for authored RL tasks (plans/neuralese/RL_TASK_AUTHORING_KIT.md section 11).

Qualification never executes a program the way a model would, so an `expected` written from intent or dataset gold
passes every replay. The round-1 audit caught every such error by solving each task from the program text and inputs
alone. This tool automates the mechanical half of that audit:

  blind_solve.py packet TASKS.jsonl -o PACKET.jsonl   strip expected/wrong answers/aliases/reference/smoke/groups
  blind_solve.py compare TASKS.jsonl ANSWERS.json     compare a solver's answers with `expected` (and folders)

PACKET rows carry `id`, `root`, `files`, `inputs`, `folders` and `returns` (the root's declared type). ANSWERS is a
JSON object `{task id: {"answer": <value>, "folders": {path: text}}}` (`folders` only for tasks with `folders`).
`compare` uses the task's own oracle kind (exact deep equality, or normalized: case, spaces, number formatting) and
the alias answers; it exits 1 if any task disagrees, listing the ids. A disagreement is for a human or an auditor to
settle by running the program literally: it can be a wrong `expected`, an ambiguous judgment or a solver slip.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HEAD = re.compile(r'^---\r?\n(.*?)\r?\n---', re.S)
RETURNS = re.compile(r'^returns:\s*(.*)$', re.M)


def read_tasks(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def root_returns(task):
    head = HEAD.match(task['files'][task['root']])
    match = RETURNS.search(head.group(1)) if head else None
    if not match:
        return None
    text = match.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def packet_row(task):
    row = {'id': task['id'], 'root': task['root'], 'files': task['files'], 'inputs': task.get('inputs', {}),
           'returns': root_returns(task)}
    if task.get('folders'):
        row['folders'] = task['folders']
    return row


def normalize(value):
    if isinstance(value, str):
        text = ' '.join(value.split()).lower()
        try:
            return float(text.replace(',', ''))
        except ValueError:
            return text
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def same(a, b, oracle):
    if oracle == 'normalized':
        return normalize(a) == normalize(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, oracle) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key], oracle) for key in a)
    return a == b


def compare(tasks, answers):
    disagreements, missing = [], []
    for task in tasks:
        given = answers.get(task['id'])
        if not isinstance(given, dict) or 'answer' not in given:
            missing.append(task['id'])
            continue
        oracle = task.get('oracle', 'exact')
        accepted = [task['expected']] + list(task.get('alias_answers') or [])
        reasons = []
        if not any(same(given['answer'], value, oracle) for value in accepted):
            reasons.append('answer')
        if task.get('folders') and task.get('expected_folders') is not None:
            solved = given.get('folders')
            if not isinstance(solved, dict) or any(
                    solved.get(path, None) is None or ' '.join(str(solved[path]).split()) != ' '.join(text.split())
                    for path, text in task['expected_folders'].items()) or set(solved) != set(task['expected_folders']):
                reasons.append('folders')
        if reasons:
            disagreements.append({'id': task['id'], 'differs': reasons, 'solver': given.get('answer'),
                                  'expected': task['expected']})
    return {'tasks': len(tasks), 'agree': len(tasks) - len(disagreements) - len(missing),
            'disagreements': disagreements, 'missing': missing}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    pack = sub.add_parser('packet')
    pack.add_argument('tasks')
    pack.add_argument('-o', '--out', required=True)
    comp = sub.add_parser('compare')
    comp.add_argument('tasks')
    comp.add_argument('answers')
    args = parser.parse_args(argv)
    tasks = read_tasks(args.tasks)
    if args.cmd == 'packet':
        Path(args.out).write_text(''.join(json.dumps(packet_row(task)) + '\n' for task in tasks))
        print(json.dumps({'packet': args.out, 'tasks': len(tasks)}))
        return 0
    result = compare(tasks, json.loads(Path(args.answers).read_text()))
    print(json.dumps(result, indent=1))
    return 1 if result['disagreements'] or result['missing'] else 0


if __name__ == '__main__':
    sys.exit(main())
