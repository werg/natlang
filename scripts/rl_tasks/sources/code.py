"""Code authoring (MBPP): the function's folder holds `app.ts` with a stub of the requested export; the call must
implement it. Oracle: the collector's executable authoring check calls the export on the upstream test inputs and
compares with the upstream expected values (Python literals read with ast.literal_eval; tuples become arrays, None
null). Only `assert f(literals) == literal` tests with JSON-representable values are used; nothing is executed at
build time. The reference writes a lookup implementation of the test table, which exercises the whole authoring
oracle path (module load, call, deep equality); negative controls are a stub and a table with one wrong value."""
from __future__ import annotations

import ast
import json
import math
import random

from ..common import SOURCES_DIR, make_case, pick

BASE = SOURCES_DIR
INSTRUCTIONS = [
    'Implement the exported function in app.ts so that it does what task describes. Python-style values map to '
    'TypeScript as: lists and tuples are arrays, dicts are objects, None is null. Keep the export name and parameters.',
    'task describes a function. Replace the stub in app.ts with a working TypeScript implementation of that '
    'function (arrays for lists or tuples, null for None), keeping its name and parameters.',
]


def _json_ok(value):
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return abs(value) < 2 ** 53
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_json_ok(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_ok(v) for k, v in value.items())
    return False


def _jsonable(value):
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def parse_tests(tests):
    name, runs = None, []
    for text in tests:
        try:
            node = ast.parse(text.strip()).body[0]
        except SyntaxError:
            return None, []
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare) or \
                len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
            return None, []
        call, expected_node = node.test.left, node.test.comparators[0]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
            return None, []
        try:
            args = [ast.literal_eval(a) for a in call.args]
            expected = ast.literal_eval(expected_node)
        except (ValueError, SyntaxError, TypeError):
            return None, []
        if not all(_json_ok(a) for a in args) or not _json_ok(expected) or isinstance(expected, float):
            return None, []
        if name not in (None, call.func.id):
            return None, []
        name = call.func.id
        runs.append({'args': _jsonable(args), 'expected': _jsonable(expected)})
    return name, runs


def _params(code, name, arity):
    try:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.FunctionDef) and node.name == name and len(node.args.args) == arity:
                return [a.arg for a in node.args.args]
    except SyntaxError:
        pass
    return [f'arg{i + 1}' for i in range(arity)]


def lookup_impl(name, params, runs):
    table = json.dumps([[r['args'], r['expected']] for r in runs])
    return ('const same = (a: any, b: any): boolean => Array.isArray(b) ? Array.isArray(a) && a.length === b.length && '
            'b.every((x: any, i: number) => same(a[i], x)) : b !== null && typeof b === "object" ? a !== null && '
            'typeof a === "object" && Object.keys(a).length === Object.keys(b).length && Object.keys(b).every(k => '
            'same(a[k], b[k])) : a === b;\n'
            f'const TABLE: [unknown[], unknown][] = {table};\n'
            f'export function {name}({", ".join(p + ": any" for p in params)}): any {{\n'
            f'  const args = [{", ".join(params)}];\n'
            f'  const hit = TABLE.find(([input]) => same(args, input));\n'
            f'  return hit ? hit[1] : null;\n}}\n')


def _perturb(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + 'x'
    if isinstance(value, list):
        return value[:-1] if value else [0]
    if value is None:
        return 0
    return None


def mbpp(seed=20261031):
    rows = []
    for split, path in (('train', 'train'), ('train', 'prompt'), ('validation', 'validation'), ('test', 'test')):
        rows += [(split, json.loads(line)) for line in open(f'{BASE}/mbpp/{path}.jsonl')]
    out = []
    for split, r in rows:
        name, runs = parse_tests(r['test_list'])
        if not name or len(runs) < 2:
            continue
        params = _params(r['code'], name, len(runs[0]['args']))
        if any(len(run['args']) != len(params) for run in runs):
            continue
        task = r.get('text') or r.get('prompt')
        task = task.replace('python function', 'function').replace('Python function', 'function')
        stub = f'export function {name}({", ".join(p + ": any" for p in params)}): any {{\n  throw new Error("not implemented");\n}}\n'
        impl = lookup_impl(name, params, runs)
        bad_runs = [dict(runs[0], expected=_perturb(runs[0]['expected']))] + runs[1:]
        write = lambda src: f'await folder.file("app.ts").writeText({json.dumps(src)}); return "done";'
        unit = f'mbpp:{r["task_id"]}'
        case = make_case(
            source='mbpp', family='implement_function', shape=f'mbpp-{len(runs)}tests', upstream='google-research-datasets/mbpp',
            upstream_id=str(r['task_id']), upstream_split='validation' if split in ('validation', 'test') else 'train',
            unit=unit, groups=[unit], root_name='implement', args={'task': 'string'}, returns='string',
            instruction=pick(INSTRUCTIONS, unit), inputs={'task': task}, expected=None, oracle='exact', level=1,
            license_='CC-BY-4.0', domain='other', slice_='folder_failure',
            reference_root=[['eval', {'code': write(impl), 'finish': True}]], revision='',
            extra_semantics={'folder_files': {'app.ts': stub},
                             'authoring': {'module': 'app.ts', 'export': name, 'runs': runs}})
        case['semantics']['files'][case['semantics']['root']] = case['semantics']['files'][case['semantics']['root']] \
            .replace('---\nargs:', '---\nkind: directory-reducer\nargs:', 1)
        case['curriculum']['external']['checks'] = {'wrong_code': [write(stub), write(lookup_impl(name, params, bad_runs))],
                                                    'alias_answers': [], 'wrong_answers': []}
        out.append(case)
    random.Random(seed).shuffle(out)
    return out
