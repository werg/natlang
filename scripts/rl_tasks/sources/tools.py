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

from ..common import make_case, pick

RAW = Path('/mnt/external/sdkb-archive/raw/agentic-20260927/xlam-function-calling-60k/xlam_function_calling_60k.json')
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
function receipt(name: string, spec: Spec, args: Record<string, unknown> = {}): string {
  const kept: Record<string, unknown> = {};
  for (const [key, raw] of Object.entries(args ?? {})) {
    const param = spec[key];
    let value: unknown = raw;
    if (param && param.kind === 'number' && typeof value === 'string' && /^[+-]?\\d+(\\.\\d+)?$/.test(value.trim())) value = Number(value);
    if (param && param.kind === 'boolean' && typeof value === 'string' && /^(true|false)$/i.test(value)) value = value.toLowerCase() === 'true';
    if (param && (value === '' || value === null || value === undefined) && param.optional) continue;
    if (param && param.default !== '' && param.default !== null && JSON.stringify(canon(value)) === JSON.stringify(canon(param.default))) continue;
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
        if not answers or any(a['name'] not in {t['name'] for t in tools} for a in answers):
            continue
        source, names = service(tools)
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
        split = 'train-carve'
        level = min(3, len(answers) - 1 + (len(tools) > 3))
        case = make_case(
            source='xlam', family='tool_calls', shape=f'xlam-{len(answers)}calls-{min(len(tools), 5)}tools',
            upstream='Salesforce/xlam-function-calling-60k', upstream_id=f'xlam-{r.get("id", index)}',
            upstream_split=split, unit=f'xlam-{r.get("id", index)}', groups=[f'xlam:{r.get("id", index)}'],
            root_name='fulfil_request', args={'request': 'string'}, returns='string[]',
            instruction=pick(INSTRUCTIONS, str(index)), inputs={'request': r['query']}, expected=None,
            oracle={'level': 'normalized', 'normalization': 'answer-list'}, level=level, license_='CC-BY-4.0',
            services={'api': source}, domain='actor', slice_='inline_placement', reference_root=reference,
            neuralese_room='', revision='26d14eb')
        case['curriculum']['external']['call_sets'] = {'service': 'api', 'names': names, 'gold': answers,
                                                       'wrong': wrong_sets, 'alias': [alias]}
        out.append(case)
    return out
