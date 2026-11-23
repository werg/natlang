#!/usr/bin/env python3
"""Blind-solve packets for authored RL tasks (plans/neuralese/RL_TASK_AUTHORING_KIT.md section 11).

Qualification never executes a program the way a model would, so an `expected` written from intent or dataset gold
passes every replay. The round-1 audit caught every such error by solving each task from the program text and inputs
alone. This tool automates the mechanical half of that audit:

  blind_solve.py packet TASKS.jsonl -o PACKET.jsonl   strip expected/wrong answers/aliases/reference/smoke/groups
  blind_solve.py compare TASKS.jsonl ANSWERS.json     compare a solver's answers with `expected` (and folders)
  blind_solve.py prompt                               the uniform solver prompt (every packet row carries its answer_format)

PACKET rows carry `id`, `root`, `files`, `inputs`, `folders` and `returns` (the root's declared type). ANSWERS is a
JSON object `{task id: {"answer": <value>, "folders": {path: text}}}` (`folders` only for tasks with `folders`).
`compare` uses the task's own oracle kind (exact deep equality, or normalized: case, spaces, number formatting) and
the alias answers; it exits 1 if any task disagrees, listing the ids. A disagreement is for a human or an auditor to
settle by running the program literally: it can be a wrong `expected`, an ambiguous judgment or a solver slip.
Verifier tasks (`oracle: verifier`) are judged by executing the solver's outcome with the task's own verifier
(`verifier_controls.mjs`), not by equality with `expected`: any outcome the verifier accepts agrees.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import authored_checks

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


ANSWER_FORMAT = ('Answer with {"answer": <the value the root returns, in its declared type>}.')
FOLDER_ANSWER_FORMAT = (
    'Answer with {"answer": <the value the root returns, in its declared type>, "folders": {"<path>": "<exact text>", ...}} '
    'where "folders" lists EVERY file that exists in the world folder after a correct run, with its full text: files the '
    'run moved appear once under their new path (the old path is gone), files it wrote appear with the text the program '
    'tells it to write, and files it never touched appear unchanged. A file that does not exist after the run must not '
    'be listed. Copy file text exactly as the program writes it (including its trailing newline when the program writes one).')

VERIFIER_NOTE = (' This task is judged by checking your outcome against what the program asks for, not against one '
                 'expected text: any outcome that does what the program means passes. Read each item for what it '
                 'means, as a careful person would, and apply the code steps exactly.')

SOLVER_PROMPT = (
    'You are the blind solver for authored natlang RL tasks. Each packet row has a root program, its helper files, '
    'concrete inputs, (for folder tasks) the starting folder, and the declared return type. Solve every task by running '
    'the program LITERALLY: simulate each helper as a reader that sees only the arguments it receives plus its own text, '
    'in the order the root calls them, apply the root\'s code steps exactly, and write what the root returns. Do not '
    'answer by intent. If a judgment has two defensible readings, give your best literal answer and add the id to '
    '"ambiguous" with a one-line reason. Each row has an "answer_format": follow it exactly. Output one JSON object '
    '{"answers": {<id>: {"answer": ..., "folders": {...}}}, "ambiguous": {<id>: <reason>}}, with "folders" only for rows '
    'that have it.')


def packet_row(task):
    row = {'id': task['id'], 'root': task['root'], 'files': task['files'], 'inputs': task.get('inputs', {}),
           'returns': root_returns(task), 'answer_format': ANSWER_FORMAT}
    if task.get('folders'):
        row['folders'] = task['folders']
        row['answer_format'] = FOLDER_ANSWER_FORMAT
    if task.get('oracle') == 'verifier':
        row['answer_format'] += VERIFIER_NOTE
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


def verifier_verdicts(tasks, answers, runtime=None):
    """{task id: {accepted, score, issues}}: each verifier task's given answer judged by its own verifier."""
    chosen = [task for task in tasks if task.get('oracle') == 'verifier']
    if not chosen:
        return {}
    runtime = runtime or os.environ.get('NATLANG_VERIFIER_RUNTIME') or str(authored_checks.REPO / 'ts-host')
    answers_path = Path(os.environ.get('TMPDIR', '/tmp')) / f'blind-solve-answers-{os.getpid()}.json'
    answers_path.write_text(json.dumps(answers))
    try:
        done = subprocess.run([authored_checks.node(), str(Path(__file__).resolve().parent / 'verifier_controls.mjs'),
                               runtime, '/dev/stdin', str(answers_path)],
                              input=''.join(json.dumps(task) + '\n' for task in chosen), capture_output=True, text=True,
                              cwd=os.environ.get('NATLANG_REPO_ROOT') or str(authored_checks.REPO))
    finally:
        answers_path.unlink(missing_ok=True)
    if done.returncode:
        raise RuntimeError('verifier_controls.mjs failed: ' + done.stderr[-2000:])
    return json.loads(done.stdout)


def compare(tasks, answers, runtime=None):
    disagreements, missing = [], []
    judged = verifier_verdicts(tasks, {key: value for key, value in answers.items() if isinstance(value, dict)}, runtime)
    for task in tasks:
        given = answers.get(task['id'])
        if not isinstance(given, dict) or 'answer' not in given:
            missing.append(task['id'])
            continue
        oracle = task.get('oracle', 'exact')
        if oracle == 'verifier':
            verdict = judged.get(task['id']) or {}
            if verdict.get('accepted') is not True:
                disagreements.append({'id': task['id'], 'differs': ['verifier'], 'solver': given.get('answer'),
                                      'verdict': verdict.get('issues') or verdict.get('error')})
            continue
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
    sub.add_parser('prompt', help='print the uniform solver prompt')
    comp = sub.add_parser('compare')
    comp.add_argument('tasks')
    comp.add_argument('answers')
    args = parser.parse_args(argv)
    if args.cmd == 'prompt':
        print(SOLVER_PROMPT)
        return 0
    tasks = read_tasks(args.tasks)
    if args.cmd == 'packet':
        Path(args.out).write_text(''.join(json.dumps(packet_row(task)) + '\n' for task in tasks))
        print(json.dumps({'packet': args.out, 'tasks': len(tasks)}))
        return 0
    given = json.loads(Path(args.answers).read_text())
    ambiguous = given.get('ambiguous', {}) if 'answers' in given else {}
    result = compare(tasks, given['answers'] if 'answers' in given else given)
    result['ambiguous'] = ambiguous
    print(json.dumps(result, indent=1))
    return 1 if result['disagreements'] or result['missing'] or ambiguous else 0


if __name__ == '__main__':
    sys.exit(main())
