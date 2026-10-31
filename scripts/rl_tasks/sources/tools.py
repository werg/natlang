"""Function calling (xLAM-60k): a user request plus an `api` service exposing the request's tools. Natlang framing:
carry out the request by calling the api functions and return what they returned. Each api function returns a
receipt naming its canonical call (declared defaults and empty optional arguments dropped, numeric strings read as
numbers for numeric parameters, keys sorted), so the returned receipts certify which calls were made. Oracle: the
multiset of receipts equals the gold calls' receipts (answer-list). The reference calls the gold functions through
the real service, so the receipts the oracle expects are the ones the service produces."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..common import ARCHIVE_RAW, SOURCES_DIR, make_case, pick, text_digest

RAW = ARCHIVE_RAW / 'agentic-20260927/xlam-function-calling-60k/xlam_function_calling_60k.json'
INSTRUCTIONS = [
    'Carry out request by calling the functions of the api service that it needs, with the arguments it specifies. '
    'Return the values those calls return, in the order you made the calls.',
    'request is a user\'s instruction. Make exactly the api calls that fulfil it (several when it asks for several '
    'things) and return the list of what each call returned.',
    'Fulfil request using the api service. Call each needed function once with arguments taken from request, then '
    'return the call results as a list.',
]


def ident(name: str) -> str:
    out = re.sub(r'[^A-Za-z0-9_]', '_', name)
    return out if re.match(r'[A-Za-z_]', out) else '_' + out


def ts_type(text: str) -> str:
    t = (text or '').split(',')[0].strip().lower()
    if t.startswith('int') or t.startswith('float'):
        return 'number'
    if t.startswith('bool'):
        return 'boolean'
    if t.startswith('str'):
        return 'string'
    if t.startswith('list[int') or t.startswith('list[float') or t.startswith('list[union[int'):
        return 'number[]'
    if t.startswith('list[str'):
        return 'string[]'
    return 'any'


SERVICE_HEAD = '''import { createHash } from 'node:crypto';
type Spec = Record<string, { kind: string; default: unknown; optional: boolean }>;
function canon(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canon((value as any)[key])]));
  return value;
}
function coerce(param: { kind: string }, value: unknown): unknown {
  if (param.kind === 'number' && typeof value === 'string' && /^[+-]?\\d+(\\.\\d+)?$/.test(value.trim())) return Number(value);
  if (param.kind === 'boolean' && typeof value === 'string' && /^(true|false)$/i.test(value)) return value.toLowerCase() === 'true';
  return value;
}
function receipt(name: string, spec: Spec, args: Record<string, unknown> = {}): string {
  const kept: Record<string, unknown> = {};
  for (const [key, raw] of Object.entries(args ?? {})) {
    const param = spec[key];
    const value = param ? coerce(param, raw) : raw;
    if (param && (value === '' || value === null || value === undefined) && param.optional) continue;
    if (param && param.default !== '' && param.default !== null && [param.default, coerce(param, param.default)]
      .some(d => JSON.stringify(canon(value)) === JSON.stringify(canon(d)))) continue;
    kept[key] = value;
  }
  const text = name + JSON.stringify(canon(kept));
  return name + '#' + createHash('sha256').update(text).digest('hex').slice(0, 12);
}
'''


def service(tools) -> tuple[str, dict]:
    parts, names = [SERVICE_HEAD], {}
    for tool in tools:
        name = ident(tool['name'])
        names[tool['name']] = name
        params = tool.get('parameters') or {}
        spec, fields = {}, []
        for key, info in params.items():
            info = info or {}
            optional = 'optional' in str(info.get('type', '')) or 'default' in info
            default = info.get('default', '')
            spec[key] = {'kind': ts_type(info.get('type', '')), 'default': default, 'optional': optional}
            doc = str(info.get('description', '')).replace('*/', '* /').replace('\n', ' ')
            default_note = f' Default: {json.dumps(default)}.' if default not in ('', None) else ''
            fields.append(f'  /** {doc}{default_note} */\n  {json.dumps(key)}{"?" if optional else ""}: '
                          f'{ts_type(info.get("type", ""))};')
        doc = str(tool.get('description', '')).replace('*/', '* /').replace('\n', ' ')
        parts.append(f'/** {doc} */\nexport function {name}(args: {{\n' + '\n'.join(fields) +
                     f'\n}}): string {{ return receipt({json.dumps(name)}, {json.dumps(spec)}, args as any); }}\n')
    return '\n'.join(parts), names


def xlam(cap_train=1500, cap_eval=150, seed=20261031):
    rows = json.load(open(RAW))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out = []
    for n, index in enumerate(order[:cap_train + 2 * cap_eval]):
        r = rows[index]
        tools, answers = json.loads(r['tools']), json.loads(r['answers'])
        case = call_set_case(source='xlam', upstream='Salesforce/xlam-function-calling-60k', item=f'xlam-{r.get("id", index)}',
                             group=f'xlam:{r.get("id", index)}', key=str(index), tools=tools, answers=answers,
                             request=r['query'], license_='CC-BY-4.0', revision='26d14eb')
        if case:
            out.append(case)
    return out


def call_set_case(*, source, upstream, item, group, key, tools, answers, request, license_, revision):
    """A request plus an `api` service of its tools (xLAM-style specs); the answer is the receipts of the gold calls."""
    if not answers or any(a['name'] not in {t['name'] for t in tools} for a in answers):
        return None
    source_code, names = service(tools)
    calls = ', '.join(f'await api.{names[a["name"]]}({json.dumps(a.get("arguments") or {})})' for a in answers)
    reference = [['eval', {'code': f'return [{calls}];', 'finish': True}]]
    # Meaningful wrong call sets: the first gold call with one argument dropped, or a sibling tool instead.
    first = answers[0]
    wrong_sets = []
    if first.get('arguments'):
        args = dict(first['arguments'])
        args.pop(sorted(args)[-1])
        wrong_sets.append([{'name': first['name'], 'arguments': args}] + answers[1:])
    sibling = next((t for t in tools if t['name'] != first['name']), None)
    if sibling:
        wrong_sets.append([{'name': sibling['name'], 'arguments': {}}] + answers[1:])
    if len(answers) > 1:
        wrong_sets.append(answers[:-1])
    if first.get('arguments'):
        k = sorted(first['arguments'])[0]
        value = first['arguments'][k]
        changed = (not value) if isinstance(value, bool) else value + 1 if isinstance(value, (int, float)) else \
            str(value) + 'x' if isinstance(value, str) else None
        if changed is not None:
            wrong_sets.append([{'name': first['name'], 'arguments': {**first['arguments'], k: changed}}] + answers[1:])
    # Alternative correct forms: numbers given as strings, explicit defaults.
    alias = []
    for a in answers:
        spec = {t['name']: t.get('parameters') or {} for t in tools}[a['name']]
        args = {k: (str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in (a.get('arguments') or {}).items()}
        for k, info in spec.items():
            if k not in args and (info or {}).get('default') not in ('', None):
                args[k] = info['default']
        alias.append({'name': a['name'], 'arguments': args})
    level = min(3, len(answers) - 1 + (len(tools) > 3))
    case = make_case(
        source=source, family='tool_calls', shape=f'{source}-{len(answers)}calls-{min(len(tools), 5)}tools',
        upstream=upstream, upstream_id=item, upstream_split='train-carve', unit=item, groups=[group],
        root_name='fulfil_request', args={'request': 'string'}, returns='string[]',
        instruction=pick(INSTRUCTIONS, key), inputs={'request': request}, expected=None,
        oracle={'level': 'normalized', 'normalization': 'answer-list'}, level=level, license_=license_,
        services={'api': source_code}, domain='actor', slice_='inline_placement', reference_root=reference,
        neuralese_room='', revision=revision)
    case['curriculum']['external']['call_sets'] = {'service': 'api', 'names': names, 'gold': answers,
                                                   'wrong': wrong_sets, 'alias': [alias]}
    return case


# --- ToolACE: single-request function calling, same receipt-certified framing as xLAM -------------------------------

TOOLACE = ARCHIVE_RAW / 'agentic-20260927/toolace/data.json'
JSON_TO_PY = {'string': 'str', 'str': 'str', 'integer': 'int', 'int': 'int', 'number': 'float', 'float': 'float',
              'double': 'float', 'boolean': 'bool', 'bool': 'bool', 'array': 'list', 'list': 'list', 'object': 'dict',
              'dict': 'dict'}


def _toolace_tools(system):
    start = system.index('[{"')
    tools, _ = json.JSONDecoder().raw_decode(system[start:])
    out = []
    for tool in tools:
        props = ((tool.get('parameters') or {}).get('properties') or {})
        required = set((tool.get('parameters') or {}).get('required') or [])
        params = {}
        for key, info in props.items():
            kind = JSON_TO_PY.get(str((info or {}).get('type', '')).lower(), 'any')
            if kind == 'list':
                item = str(((info or {}).get('items') or {}).get('type', '')).lower()
                kind = {'string': 'List[str]', 'integer': 'List[int]', 'number': 'List[float]'}.get(item, 'list')
            entry = {'type': kind + ('' if key in required else ', optional'),
                     'description': (info or {}).get('description', '')}
            if 'default' in (info or {}):
                entry['default'] = info['default']
            params[key] = entry
        out.append({'name': tool['name'], 'description': tool.get('description', ''), 'parameters': params})
    return out


def _split_calls(text):
    """`[A(x=1), B C(y="a, b")]` -> ['A(x=1)', 'B C(y="a, b")'], aware of quotes and nesting."""
    text = text.strip()
    if not (text.startswith('[') and text.endswith(']')):
        return None
    inner, parts, depth, quote, current = text[1:-1], [], 0, None, ''
    for i, ch in enumerate(inner):
        if quote:
            current += ch
            if ch == quote and inner[i - 1] != '\\':
                quote = None
            continue
        if ch in '"\'':
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
            continue
        current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_call(text):
    import ast
    if '(' not in text or not text.endswith(')'):
        return None
    name, rest = text.split('(', 1)
    try:
        node = ast.parse('f(' + rest, mode='eval').body
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    except (SyntaxError, ValueError):
        return None
    if node.args or any(k is None for k in args):
        return None
    return {'name': name.strip(), 'arguments': args}


def toolace(cap=1800, seed=20261031):
    rows = json.load(open(TOOLACE))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out = []
    for index in order:
        r = rows[index]
        conv = r['conversations']
        if len(conv) != 2 or conv[0]['from'] != 'user' or conv[1]['from'] != 'assistant':
            continue
        texts = _split_calls(conv[1]['value'])
        if not texts:
            continue
        answers = [_parse_call(t) for t in texts]
        if any(a is None for a in answers):
            continue
        try:
            tools = _toolace_tools(r['system'])
        except (ValueError, json.JSONDecodeError):
            continue
        item = f'toolace-{index}'
        case = call_set_case(source='toolace', upstream='Team-ACE/ToolACE', item=item, group=f'toolace:{index}',
                             key=item, tools=tools, answers=answers, request=conv[0]['value'],
                             license_='Apache-2.0', revision='6bda777')
        if case:
            out.append(case)
        if len(out) >= cap:
            break
    return out


# --- NESTFUL: nested calls with data flow, over its executable arithmetic functions --------------------------------

NESTFUL = SOURCES_DIR + '/nestful/data__train.jsonl'
# NESTFUL's basic_functions.py (IBM/NESTFUL@fc2c412, data_v2/executable_functions), ported with Python semantics:
# floor division and modulo follow the divisor's sign, sqrt is the integer square root.
NESTFUL_IMPL = {
    'add': '(a, b) => a + b', 'subtract': '(a, b) => a - b', 'multiply': '(a, b) => a * b', 'divide': '(a, b) => a / b',
    'power': '(a, b) => a ** b', 'inverse': '(a) => 1 / a', 'negate': '(a) => -a', 'negate_prob': '(p) => 1 - p',
    'sqrt': '(a) => isqrt(a)', 'floor': '(a) => Math.floor(a)', 'reminder': '(a, b) => pymod(a, b)',
    'remainder': '(a, b) => pymod(a, b)', 'lcm': '(a, b) => Math.abs(a * b) / gcd(a, b)', 'gcd': '(a, b) => gcd(Math.trunc(a), Math.trunc(b))',
    'factorial': '(n) => factorial(n)', 'choose': '(n, r) => factorial(n) / (factorial(r) * factorial(n - r))',
    'permutation': '(n, r) => factorial(n) / factorial(n - r)', 'rectangle_area': '(l, w) => l * w',
    'rectangle_perimeter': '(l, w) => 2 * (l + w)', 'square_area': '(s) => s * s', 'square_perimeter': '(s) => 4 * s',
    'square_edge_by_area': '(a) => Math.sqrt(a)', 'square_edge_by_perimeter': '(p) => p / 4',
    'circle_area': '(r) => Math.PI * r ** 2', 'circumface': '(r) => 2 * Math.PI * r', 'surface_sphere': '(r) => 4 * Math.PI * r ** 2',
    'volume_sphere': '(r) => 4 / 3 * Math.PI * r ** 3', 'volume_cylinder': '(r, h) => Math.PI * r ** 2 * h',
    'surface_cylinder': '(r, h) => 2 * Math.PI * r * (r + h)', 'volume_cone': '(r, h) => 1 / 3 * Math.PI * r ** 2 * h',
    'volume_cube': '(s) => s ** 3', 'surface_cube': '(s) => 6 * s ** 2', 'cube_edge_by_volume': '(v) => Math.pow(v, 1 / 3)',
    'diagonal': '(l, w) => Math.sqrt(l ** 2 + w ** 2)', 'triangle_area': '(b, h) => 0.5 * b * h',
    'rhombus_area': '(d1, d2) => 0.5 * d1 * d2', 'speed': '(d, t) => d / t', 'max_number': '(xs) => Math.max(...xs)',
    'min_number': '(xs) => Math.min(...xs)', 'log': '(x, base) => Math.trunc(Math.log(x) / Math.log(base))',
}
NESTFUL_HEAD = '''function isqrt(a: number): number {
  if (!Number.isInteger(a) || a < 0) throw new Error('sqrt needs a non-negative integer');
  let r = Math.floor(Math.sqrt(a)); while (r * r > a) r--; while ((r + 1) * (r + 1) <= a) r++; return r;
}
function pymod(a: number, b: number): number { const m = a % b; return m !== 0 && (m < 0) !== (b < 0) ? m + b : m; }
function gcd(a: number, b: number): number { a = Math.abs(a); b = Math.abs(b); while (b) [a, b] = [b, a % b]; return a; }
function factorial(n: number): number {
  if (!Number.isInteger(n) || n < 0) throw new Error('factorial needs a non-negative integer');
  let out = 1; for (let i = 2; i <= n; i++) out *= i; return out;
}
'''
NESTFUL_INSTRUCTIONS = [
    'Answer question by calling the functions of the api service, feeding the result of one call into the next where '
    'needed. Each function returns { result }. Return the final numeric result.',
    'question is a word problem. Solve it as a chain of api calls (each returns { result }), using earlier results as '
    'later arguments, and return the final number.',
]


def _py_eval(name, args):
    import math as m
    fns = {'add': lambda a, b: a + b, 'subtract': lambda a, b: a - b, 'multiply': lambda a, b: a * b,
           'divide': lambda a, b: a / b, 'power': lambda a, b: a ** b, 'inverse': lambda a: 1 / a,
           'negate': lambda a: -a, 'negate_prob': lambda p: 1 - p, 'sqrt': lambda a: m.isqrt(a),
           'floor': lambda a: m.floor(a), 'reminder': lambda a, b: a % b, 'remainder': lambda a, b: a % b,
           'lcm': lambda a, b: abs(a * b) // m.gcd(a, b), 'gcd': lambda a, b: m.gcd(int(a), int(b)),
           'factorial': lambda n: m.factorial(n), 'choose': lambda n, r: m.comb(n, r),
           'permutation': lambda n, r: m.factorial(n) // m.factorial(n - r), 'rectangle_area': lambda l, w: l * w,
           'rectangle_perimeter': lambda l, w: 2 * (l + w), 'square_area': lambda s: s * s,
           'square_perimeter': lambda s: 4 * s, 'square_edge_by_area': lambda a: m.sqrt(a),
           'square_edge_by_perimeter': lambda p: p / 4, 'circle_area': lambda r: m.pi * r ** 2,
           'circumface': lambda r: 2 * m.pi * r, 'surface_sphere': lambda r: 4 * m.pi * r ** 2,
           'volume_sphere': lambda r: 4 / 3 * m.pi * r ** 3, 'volume_cylinder': lambda r, h: m.pi * r ** 2 * h,
           'surface_cylinder': lambda r, h: 2 * m.pi * r * (r + h), 'volume_cone': lambda r, h: 1 / 3 * m.pi * r ** 2 * h,
           'volume_cube': lambda s: s ** 3, 'surface_cube': lambda s: 6 * s ** 2, 'cube_edge_by_volume': lambda v: m.pow(v, 1 / 3),
           'diagonal': lambda l, w: m.sqrt(l ** 2 + w ** 2), 'triangle_area': lambda b, h: 0.5 * b * h,
           'rhombus_area': lambda a, b: 0.5 * a * b, 'speed': lambda d, t: d / t, 'max_number': lambda xs: max(xs),
           'min_number': lambda xs: min(xs), 'log': lambda x, b: int(m.log(x, b))}
    return fns[name](*args)


def _nestful_run(calls):
    """Execute a gold call sequence; `$var_k.result$` arguments read earlier results. Returns the last result."""
    results, last = {}, None
    for call in calls:
        args = []
        for value in (call.get('arguments') or {}).values():
            if isinstance(value, str) and re.fullmatch(r'\$var_?\d+\.result\$', value):
                value = results[value[:-len('.result$')]]
            args.append(value)
        last = _py_eval(call['name'], args)
        if isinstance(last, complex):
            raise ValueError('complex result')
        results[call['label']] = last
    return last


def nestful(cap=1800, seed=20261031):
    rows = [json.loads(line) for line in open(NESTFUL)]
    random.Random(seed).shuffle(rows)
    out, mismatched = [], 0
    for r in rows:
        calls, tools = json.loads(r['output']), json.loads(r['tools'])
        if not calls or any(c['name'] not in NESTFUL_IMPL for c in calls):
            continue
        try:
            value = float(_nestful_run(calls))
            gold = float(r['gold_answer'])
        except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError):
            continue
        if abs(value - gold) > 1e-6 * max(1.0, abs(gold)):
            mismatched += 1
            continue
        offered = [t for t in tools if t['name'] in NESTFUL_IMPL]
        parts = [NESTFUL_HEAD]
        for tool in offered:
            params = list((tool.get('parameters') or {}).items())
            doc = str(tool.get('description', '')).replace('*/', '* /')
            fields = '\n'.join(f'  /** {str((info or {}).get("description", "")).replace("*/", "* /")} */\n  {key}: '
                               f'{"number[]" if "list" in str((info or {}).get("type", "")).lower() else "number"};'
                               for key, info in params)
            call_args = ', '.join(f'args.{key}' for key, _ in params)
            parts.append(f'/** {doc} */\nexport function {tool["name"]}(args: {{\n{fields}\n}}): {{ result: number }} '
                         f'{{ return {{ result: ({NESTFUL_IMPL[tool["name"]]})({call_args}) }}; }}\n')
        steps, lines = {}, []
        for call in calls:
            args = ', '.join(f'{k}: {steps[v[:-len(".result$")]] + ".result" if isinstance(v, str) and v.endswith(".result$") else json.dumps(v)}'
                             for k, v in (call.get('arguments') or {}).items())
            var = f'v{len(steps) + 1}'
            steps[call['label']] = var
            lines.append(f'const {var} = await api.{call["name"]}({{ {args} }});')
        lines.append(f'return {var}.result;')
        reference = [['eval', {'code': '\n'.join(lines), 'finish': True}]]
        value = round(value, 9)
        item = f'nestful-{r["sample_id"]}'
        out.append(make_case(
            source='nestful', family='nested_calls', shape=f'nestful-{min(len(calls), 6)}calls',
            upstream='ibm-research/nestful', upstream_id=item, upstream_split='train-carve', unit=item,
            groups=[item, f'mathqa-problem:{text_digest(r["input"])[:16]}'], root_name='solve_with_api',
            args={'question': 'string'}, returns='number', instruction=pick(NESTFUL_INSTRUCTIONS, item),
            inputs={'question': r['input']}, expected=value,
            oracle={'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-4},
            level=min(3, max(1, len(calls) // 2)), license_='Apache-2.0', services={'api': '\n'.join(parts)},
            domain='actor', slice_='inline_placement', reference_root=reference,
            wrong=[value * 1.5 + 1, -value - 1], aliases=[str(value)], revision='f1ed646'))
        if len(out) >= cap:
            break
    print(json.dumps({'nestful_gold_mismatch': mismatched}))
    return out
