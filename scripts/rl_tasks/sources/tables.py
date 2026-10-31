"""Table question answering: WikiTableQuestions (answer lists) and DataBench (typed answers over real data tables).
Natlang framing: the table is an input value (`{ columns, rows }`) or, for DataBench's full tables, a `table` service
paging over an asset file bound by SHA-256; questions need filtering, counting and aggregation in eval.
Oracles: answer-list (unordered short answers) for list answers, numeric tolerance 1e-6 for numbers, normalized exact
for booleans and categories."""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import random
from pathlib import Path

from ..common import BUILD_ASSETS, SOURCES_DIR, TASK_ASSETS, distinct_wrong, make_case, pick

BASE = SOURCES_DIR
ASSET_ROOT = TASK_ASSETS

WTQ_INSTRUCTIONS = [
    'Answer question from table. Return the answer as a list of values (one item for a single answer), each copied '
    'from the table or computed from it, such as a count.',
    'table is a table from a web page. Find, count or compare what question asks about and return the answer values '
    'as a list.',
]


def wtq(cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('test', cap_eval * 2)):
        rows = [json.loads(line) for line in open(f'{BASE}/wikitablequestions/{split}.jsonl')]
        random.Random(seed).shuffle(rows)
        pool = [r['answers'] for r in rows[:2000]]
        for r in rows[:cap]:
            header, body = r['table'][0], r['table'][1:]
            if not body or len(json.dumps(body)) > 60000:
                continue
            unit = f'wtq-table:{r["table_csv"]}'
            answers = list(r['answers'])
            out.append(make_case(
                source='wikitablequestions', family='table_qa', shape=f'wtq-{min(len(answers), 3)}ans',
                upstream='stanfordnlp/wikitablequestions', upstream_id=r['id'],
                upstream_split='validation' if split == 'test' else 'train', unit=unit, groups=[unit],
                root_name='answer_from_table', args={'question': 'string', 'table': '{ columns: string[]; rows: string[][] }'},
                returns='string[]', instruction=pick(WTQ_INSTRUCTIONS, r['id']),
                inputs={'question': r['question'], 'table': {'columns': header, 'rows': body}}, expected=answers,
                oracle={'level': 'normalized', 'normalization': 'answer-list'}, level=1 + (len(body) > 20),
                license_='CC-BY-SA-4.0', domain='relational', slice_='inline_placement',
                wrong=distinct_wrong(answers, pool, 2) + ([answers[:-1]] if len(answers) > 1 else []),
                aliases=[list(reversed(answers)), [a.upper() for a in answers]], revision=''))
    return out


DB_INSTRUCTIONS = [
    'Answer question about the data in table. Compute the answer exactly in eval over all rows; do not estimate. '
    'Return a value of the declared type.',
    'table holds a real dataset. Filter, count, rank or aggregate its rows in code to answer question, and return the '
    'answer with the declared type.',
]
DB_SERVICE_NOTE = '\n\nThe table service gives its column names and pages of rows; go through every page.'


def _typed(kind, text):
    if kind == 'boolean':
        return 'boolean', text.strip().lower() == 'true', 'exact'
    if kind == 'number':
        value = float(text)
        if value.is_integer():
            return 'number', int(value), 'normalized'
        return 'number', value, {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-6}
    if kind == 'category':
        return 'string', text, 'normalized'
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        raise ValueError('list answer is not a list')
    if kind == 'list[number]':
        value = [float(v) for v in value]
        return 'number[]', [int(v) if v.is_integer() else v for v in value], {'level': 'normalized', 'normalization': 'answer-list'}
    return 'string[]', [str(v) for v in value], {'level': 'normalized', 'normalization': 'answer-list'}


def _wrong(kind, value):
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, (int, float)):
        return [value + 1 if isinstance(value, int) else value * 1.01 + 0.5, value * 2 + 3]
    if isinstance(value, list):
        return [value[:-1] if len(value) > 1 else value + value[:1]]
    return []


def _clean(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, 'isoformat'):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def table_service(file, digest):
    return f'''import {{ createHash }} from 'node:crypto';
import {{ readFileSync }} from 'node:fs';
const FILE = {json.dumps(file)};
const SHA256 = {json.dumps(digest)};
function load(): {{ columns: string[]; rows: unknown[][] }} {{
  const cache = ((globalThis as any).__natlangRlTables ??= new Map());
  if (!cache.has(SHA256)) {{
    const bytes = readFileSync((process.env.NATLANG_RL_TASK_ASSETS ?? {json.dumps(ASSET_ROOT)}) + '/' + FILE);
    if (createHash('sha256').update(bytes).digest('hex') !== SHA256) throw new Error('table asset changed: ' + FILE);
    cache.set(SHA256, JSON.parse(bytes.toString('utf8')));
  }}
  return cache.get(SHA256);
}}
/** Column names, in row order. */
export function columns(): string[] {{ return load().columns; }}
/** Number of rows. */
export function count(): number {{ return load().rows.length; }}
/** Rows start .. start + size - 1 (at most 500), each an array of values in column order. */
export function rows(start: number, size: number): unknown[][] {{ return load().rows.slice(start, start + Math.min(size, 500)); }}
'''


def databench(out_assets=BUILD_ASSETS, max_full_rows=20000, seed=20261031):
    import pyarrow.parquet as pq
    out = []
    for directory in sorted(glob.glob(f'{BASE}/databench/files/data/*_*')):
        name = os.path.basename(directory)
        if not os.path.exists(f'{directory}/qa.parquet'):
            continue
        qa = pq.read_table(f'{directory}/qa.parquet').to_pylist()
        sample = pq.read_table(f'{directory}/sample.parquet')
        columns = sample.schema.names
        sample_rows = [[_clean(v) for v in row.values()] for row in sample.to_pylist()]
        full = pq.read_table(f'{directory}/all.parquet')
        service = None
        if full.num_rows <= max_full_rows:
            rel = f'tables/databench-{name}.json'
            target = Path(out_assets) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                data = {'columns': full.schema.names, 'rows': [[_clean(v) for v in row.values()] for row in full.to_pylist()]}
                target.write_text(json.dumps(data, ensure_ascii=False, default=str))
            service = table_service(rel, hashlib.sha256(target.read_bytes()).hexdigest())
        unit = f'databench:{name}'
        for index, q in enumerate(qa):
            for variant in ('sample', 'full'):
                text = q['sample_answer'] if variant == 'sample' else q['answer']
                if variant == 'full' and service is None:
                    continue
                try:
                    returns, value, oracle = _typed(q['type'], str(text))
                except (ValueError, SyntaxError):
                    continue
                args = {'question': 'string'}
                inputs = {'question': q['question']}
                instruction = pick(DB_INSTRUCTIONS, f'{unit}:{index}')
                if variant == 'sample':
                    args['table'] = '{ columns: string[]; rows: unknown[][] }'
                    inputs['table'] = {'columns': columns, 'rows': sample_rows}
                else:
                    instruction += DB_SERVICE_NOTE
                out.append(make_case(
                    source='databench', family=f'data_analysis_{q["type"].replace("[", "_").replace("]", "")}',
                    shape=f'databench-{variant}-{q["type"]}', upstream='cardiffnlp/databench',
                    upstream_id=f'{name}:{index}:{variant}', upstream_split='train-carve', unit=unit, groups=[unit],
                    root_name='analyze_table', args=args, returns=returns, instruction=instruction, inputs=inputs,
                    expected=value, oracle=oracle, level=1 if variant == 'sample' else 2, license_='MIT',
                    services={'table': service} if variant == 'full' else None, domain='relational',
                    slice_='inline_placement', wrong=_wrong(q['type'], value), aliases=[], revision='e75d53a'))
    return out
