#!/usr/bin/env python3
"""Static construct and contract checks for authored RL tasks (plans/neuralese/RL_TASK_AUTHORING_KIT.md section 4).

Qualification replays the reference, the negatives and the scripted smoke; none of that executes the program the way
a model would, so a task can pass every replay while its construct tags, folders, readouts, skills or helper calls
are fiction. These checks read the authored task and its program files and reject (errors) or flag (warnings):

- ``folder_reducer`` without ``folders`` (the root must reduce the task's world folder, not a folder it builds), and a
  ``kind: directory-reducer`` root without one;
- a ``readout: decision`` helper that calls helpers or owns a callable folder: readouts run in one pass with no tools,
  so sub-helper calls inside them never run;
- ``skills`` files bound to a function that does not exist, or read by a function other than the one owning the
  folder (the runtime binds ``<folder>/skills/<name>/SKILL.md`` only to the call that owns ``<folder>``);
- ``multi_operation_codebase`` on a world whose tasks run a single root operation, or whose roots share no helper file
  (same name, identical text, in the folders of two or more roots) (batch-level);
- ``iterateOn`` without an explicit stopping condition in its instructions, and an ``iterate`` tag without ``iterateOn``;
- ``call_depth_3`` without a root -> helper -> sub-helper chain, or whose middle helper never calls its sub-helper;
  warning when the middle helper looks like a pass-through or the sub-helper like a one-line comparison;
- helper calls spelled out in the instructions (backticked code) with a different number of arguments than the
  helper declares.

The call_depth_3 warnings are heuristics: pass-through means identical argument and return types to its sub-helper, thin
means under ``THIN_WORDS`` words of its own besides the call; they point the auditor at a task, they do not decide it.
Smoke and reference scripts get the same argument-count check in structure_smoke.mjs (it parses the code).

  authored_checks.py TASKS.jsonl     print {task id: {errors, warnings}} for every task with findings
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

THIN_WORDS = 20
NEURALESE_CONSTRUCTS = frozenset({'typed_neuralese', 'neuralese_relay', 'neuralese_fan_in', 'iterate_neuralese'})
ONE_LINE_WORDS = 25
STOP_WORD = re.compile(r'\b(stop|stops|stopping|stopped|until|halt|halts|terminate|terminates|ends? when)\b', re.I)
CODE = re.compile(r'```.*?```|`[^`\n]+`', re.S)
CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
SKILL_PATH = re.compile(r'^(.*)/skills/([^/]+)/')
HEAD = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n?(.*)$', re.S)


def split_file(text):
    match = HEAD.match(text)
    return (match.group(1), match.group(2)) if match else ('', text)


def split_top(text, separators):
    """Split at separators that are outside brackets, braces, parentheses, angle brackets and quotes."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == '\\':
                i += 1
            elif char == quote:
                quote = None
        elif char in '"\'`':
            quote = char
        elif char in '([{<':
            depth += 1
        elif char in ')]}>':
            if not (char == '>' and text[i - 1:i] == '='):
                depth -= 1
        elif char in separators and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def declared_args(header):
    """(required, total) argument counts from the frontmatter ``args`` field (block or inline form)."""
    lines = header.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r'^args:(.*)$', line)
        if not match:
            continue
        inline = match.group(1).strip()
        block = []
        while index + 1 < len(lines) and re.match(r'^(\s|$)', lines[index + 1]):
            index += 1
            block.append(lines[index])
        text = inline + '\n' + '\n'.join(block)
        if inline.startswith('{'):
            members = split_top(inline.strip()[1:-1] if inline.strip().endswith('}') else inline[1:], ',;')
        else:
            indent = min((len(b) - len(b.lstrip()) for b in block if b.strip()), default=0)
            members = [b.strip() for b in block if b.strip() and len(b) - len(b.lstrip()) == indent]
        names = [re.match(r'^([A-Za-z_$][\w$]*)(\?)?\s*:', member) for member in members]
        names = [m for m in names if m]
        return sum(1 for m in names if not m.group(2)), len(names)
    return 0, 0


def call_arguments(text, open_paren):
    """The argument strings of the call whose '(' is at open_paren, or None when it is not closed."""
    depth, quote, i = 0, None, open_paren
    while i < len(text):
        char = text[i]
        if quote:
            if char == '\\':
                i += 1
            elif char == quote:
                quote = None
        elif char in '"\'`':
            quote = char
        elif char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
            if depth == 0:
                return split_top(text[open_paren + 1:i], ',')
        i += 1
    return None


def arity_problems(code, children):
    """Calls to direct helpers (``children``: name -> (required, total)) with the wrong number of arguments."""
    problems = []
    for match in CALL.finditer(code):
        name = match.group(1)
        arguments = call_arguments(code, match.end() - 1)
        if arguments is None:
            continue
        if name == 'decide' and arguments and arguments[0] in children:
            name, arguments = arguments[0], arguments[1:]
        if name not in children:
            continue
        required, total = children[name]
        if not required <= len(arguments) <= total:
            problems.append(f'{name} is called with {len(arguments)} argument(s) but declares '
                            + (f'{total}' if required == total else f'{required} to {total}'))
    return problems


def functions_of(task):
    """path -> (header, body) for every .nl file."""
    return {path: split_file(text) for path, text in task['files'].items() if path.endswith('.nl')}


def children_of(functions, path):
    """Direct helpers of the function at path: name -> declared (required, total) argument counts."""
    folder = path[:-3] + '/'
    return {other[len(folder):-3]: declared_args(functions[other][0]) for other in functions
            if other.startswith(folder) and '/' not in other[len(folder):]}


def lint_task(task):
    errors, warnings = [], []
    constructs = set(task.get('constructs') or [])
    functions = functions_of(task)
    root = task['root']
    folders = task.get('folders')
    root_header = functions.get(root, ('', ''))[0]
    reducer_root = bool(re.search(r'^kind:\s*directory-reducer\s*$', root_header, re.M))

    if 'folder_reducer' in constructs and not folders:
        errors.append('folder_reducer needs `folders` (path -> file text): the root reduces the task\'s world folder, '
                      'it must not create its own items')
    if reducer_root and not folders:
        errors.append('a `kind: directory-reducer` root needs `folders`')

    for path, (header, body) in functions.items():
        name = path.rsplit('/', 1)[-1][:-3]
        if re.search(r'^readout:\s*decision', header, re.M):
            others = {p.rsplit('/', 1)[-1][:-3] for p in functions if p != path}
            called = sorted({n for n in CALL.findall(body) if n in others})
            if called:
                errors.append(f'readout helper {path} calls {", ".join(called)}: a decision readout is scored in '
                              'one pass with no tools, so calls inside it never run')
            if any(other.startswith(path[:-3] + '/') for other in task['files']):
                errors.append(f'readout helper {path} owns a callable folder: a readout is a leaf')
        if 'iterateOn' in body and not STOP_WORD.search(body):
            errors.append(f'{path} uses iterateOn without an explicit stopping condition (say when to stop)')
        for code in CODE.findall(body):
            for problem in arity_problems(code, children_of(functions, path)):
                errors.append(f'{path}: {problem}')

    smoke = task.get('smoke') or {}
    if constructs & NEURALESE_CONSTRUCTS and not smoke.get('replay'):
        errors.append('Neuralese constructs need `smoke.replay: true`: the smoke code is then the whole root script '
                      '(helpers answer with scripted values) and its returned value is judged by the oracle, so the '
                      'exact steps behind `expected` are checked by a real run, not only by the blind solve')
    if smoke.get('replay'):
        tail = re.search(r'return\s+([^;]*?);?\s*$', str(smoke.get('code', '')).strip(), re.S)
        if not tail:
            errors.append('smoke.replay code must end with `return <answer>`')
        elif re.fullmatch(r'"[^"]*"|\'[^\']*\'|-?\d[\d.]*|true|false|null', tail.group(1).strip()):
            warnings.append('heuristic: smoke.replay returns a bare literal: the replay checks no code step; compute '
                            'the answer from the helpers\' scripted values')
    if {'iterate', 'iterate_neuralese'} & constructs and not any('iterateOn' in b for _, b in functions.values()):
        errors.append('iterate/iterate_neuralese tag but no function uses iterateOn')

    for path in task['files']:
        match = SKILL_PATH.match(path)
        if not match:
            continue
        folder, skill = match.groups()
        owner = folder + '.nl'
        if owner not in functions:
            errors.append(f'skill {path}: no function {owner} owns folder {folder}/, so no call can see it')
            continue
        readers = sorted(p for p, (_, body) in functions.items()
                         if p != owner and re.search(rf'skills[./]{re.escape(skill)}\b', body))
        if readers:
            errors.append(f'skill {skill} is bound to {owner} (the function owning {folder}/) but read by '
                          f'{", ".join(readers)}; put it in the folder of the function that reads it')

    if 'call_depth_3' in constructs:
        stem = root[:-3]
        middles = [(p[:p.rindex('/')] + '.nl', p) for p in functions if p.startswith(stem + '/') and p.count('/') >= 2]
        middles = [(m, s) for m, s in middles if m in functions and m != root]
        if not middles:
            errors.append('call_depth_3 needs a root -> helper -> sub-helper chain (a file two folders below the root)')
        for middle, sub in middles:
            sub_name = sub.rsplit('/', 1)[-1][:-3]
            (m_header, m_body), (s_header, s_body) = functions[middle], functions[sub]
            if not re.search(rf'\b{re.escape(sub_name)}\b', m_body):
                errors.append(f'call_depth_3: {middle} never calls its sub-helper {sub_name}')
                continue
            own = len(re.sub(rf'[^.\n]*\b{re.escape(sub_name)}\b[^.\n]*', ' ', m_body).split())
            same = (declared_args(m_header) == declared_args(s_header)
                    and re.sub(r'\s+', '', re.sub(r'^.*?returns:', '', m_header, flags=re.S))
                    == re.sub(r'\s+', '', re.sub(r'^.*?returns:', '', s_header, flags=re.S)))
            if same:
                warnings.append(f'heuristic: {middle} has the same arguments and return type as {sub_name}: '
                                'a pass-through is not a distinct subproblem')
            elif own < THIN_WORDS:
                warnings.append(f'heuristic: {middle} adds under {THIN_WORDS} words of its own work around '
                                f'{sub_name}: does it own a different subproblem?')
            if (re.search(r'returns:\s*"?boolean', s_header) or re.search(r'readout:\s*decision', s_header)) \
                    and len(s_body.split()) < ONE_LINE_WORDS:
                warnings.append(f'heuristic: sub-helper {sub_name} is a short yes/no comparison: exact comparisons '
                                'belong in code, not in a helper')
    return errors, warnings


def helper_signature(task):
    return {(path.rsplit('/', 1)[-1], re.sub(r'\s+', ' ', text)) for path, text in task['files'].items()
            if path.endswith('.nl') and path != task['root'] and '/' in path}


def lint_batch(tasks):
    """{task id: {'errors': [...], 'warnings': [...]}} for every task, with the batch-level world checks."""
    report = {}
    for task in tasks:
        errors, warnings = lint_task(task)
        report[task['id']] = {'errors': errors, 'warnings': warnings}
    worlds = collections.defaultdict(list)
    for task in tasks:
        if 'multi_operation_codebase' in (task.get('constructs') or []):
            worlds[task['world']].append(task)
    for world, group in worlds.items():
        roots = {task['root'] for task in group}
        if len(roots) < 2:
            for task in group:
                report[task['id']]['errors'].append(
                    f'multi_operation_codebase: world {world} has one root operation ({task["root"]}) among its tagged '
                    'tasks; several root operations over shared helper files are required')
            continue
        by_helper = collections.Counter(sig for root in roots
                                        for sig in set().union(*(helper_signature(t) for t in group if t['root'] == root)))
        if not any(count >= 2 for count in by_helper.values()):
            for task in group:
                report[task['id']]['errors'].append(
                    f'multi_operation_codebase: world {world} has {len(roots)} roots but no helper file (same name, identical '
                    'text) in the folders of two or more of them; one reusable program means shared helper files')
    return report


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit(__doc__)
    tasks = [json.loads(line) for line in Path(argv[0]).read_text().splitlines() if line.strip()]
    findings = {k: v for k, v in lint_batch(tasks).items() if v['errors'] or v['warnings']}
    print(json.dumps(findings, indent=1))
    return 1 if any(v['errors'] for v in findings.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
