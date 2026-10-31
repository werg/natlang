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


# --- v3 additions: TabFact, WikiSQL, MSR-SQA ------------------------------------------------------------------------

def table_content_group(header, rows) -> str:
    """Content identity of a table across sources (whitespace/case-normalized cells), so the same web table in two
    sources stays on one split."""
    norm = lambda cell: ' '.join(str(cell).lower().split())
    text = '\n'.join('\t'.join(norm(c) for c in row) for row in [header, *rows])
    return 'table-content:' + hashlib.sha256(text.encode()).hexdigest()[:20]


def _sample_lines(path, cap, seed, keep=lambda row: True):
    """Seeded reservoir sample of `cap` eligible rows, streaming the file once (memory bounded by `cap` rows)."""
    rng, sample, seen = random.Random(seed), [], 0
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if not keep(row):
                continue
            seen += 1
            if len(sample) < cap:
                sample.append(row)
            else:
                j = rng.randrange(seen)
                if j < cap:
                    sample[j] = row
    rng.shuffle(sample)
    return sample


TABFACT_INSTRUCTIONS = [
    'Decide whether statement is true according to table. Return true when the table supports it and false when the '
    'table contradicts it. Check counts, comparisons and superlatives in eval over the rows.',
    'statement makes a claim about table (a table from a web page about caption). Verify the claim against the rows, '
    'computing any count or comparison in code, and return whether it holds.',
]
TABFACT_NEURALESE = {
    'helper': 'read_rows', 'args': {'claim': 'string', 'columns': 'string[]', 'rows': 'string[][]'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read rows (cells in columns order) for everything claim depends on, and write a private note of '
                   'the relevant cells and whether they agree with claim.',
    'root_note': 'read_rows gives a private note on a slice of the rows. You may split a long table into slices, read '
                 'each, and decide from the notes together with your own checks in eval.',
}


def tabfact(cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('validation', cap_eval), ('test', cap_eval)):
        path = f'{BASE}/tabfact/tab_fact__{split}.jsonl'
        for r in _sample_lines(path, cap * 3, seed):
            lines = [line.split('#') for line in r['table_text'].strip('\n').split('\n')]
            header, body = lines[0], lines[1:]
            if not body or any(len(row) != len(header) for row in body) or len(r['table_text']) > 20000:
                continue
            label = bool(r['label'])
            unit = f'tabfact-table:{r["table_id"]}'
            item = f'{r["table_id"]}:{hashlib.sha256(r["statement"].encode()).hexdigest()[:12]}'
            out.append(make_case(
                source='tabfact', family='table_fact_check', shape=f'tabfact-{"long" if len(body) > 15 else "short"}',
                upstream='wenhu/tab_fact', upstream_id=item, upstream_split='validation' if split != 'train' else 'train',
                unit=unit, groups=[unit, table_content_group(header, body)], root_name='check_statement',
                args={'statement': 'string', 'caption': 'string', 'table': '{ columns: string[]; rows: string[][] }'},
                returns='boolean', instruction=pick(TABFACT_INSTRUCTIONS, item),
                inputs={'statement': r['statement'], 'caption': r['table_caption'],
                        'table': {'columns': header, 'rows': body}},
                expected=label, oracle='exact', level=1 + (len(body) > 15), license_='CC-BY-4.0', domain='relational',
                slice_='inline_placement', wrong=[not label], aliases=[], revision='refs/convert/parquet',
                neuralese_room='row slices read by children, fan-in to the verdict', neuralese=TABFACT_NEURALESE))
    # bound per split after filtering
    by_split, kept = {}, []
    for case in out:
        key = case['split']
        limit = cap_train if key == 'train' else cap_eval
        if by_split.get(key, 0) < limit:
            by_split[key] = by_split.get(key, 0) + 1
            kept.append(case)
    return kept


WIKISQL_AGG = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
WIKISQL_OPS = ['=', '>', '<']
WIKISQL_INSTRUCTIONS = [
    'Answer question from table by filtering and aggregating its rows exactly in eval. Return the matching cell '
    'values as a list (one item when there is one match).',
    'question asks for values from table, possibly a count, sum, average, minimum or maximum. Compute it in code over '
    'all rows and return the answer as a list of values.',
]


def _num(cell):
    try:
        return float(str(cell).replace(',', '').strip())
    except ValueError:
        return None


def wikisql_execute(table, sql):
    """WikiSQL's gold query over its table: equality is case-insensitive on text, `>`/`<` compare numbers (an item
    whose comparison column is not numeric is skipped). Returns the answer list or None when the item is unusable."""
    header, types, rows = table['header'], table['types'], table['rows']
    conds = list(zip(sql['conds']['column_index'], sql['conds']['operator_index'], sql['conds']['condition']))
    matched = []
    for row in rows:
        ok = True
        for col, op, value in conds:
            cell = row[col]
            if op == 0:
                a, b = _num(cell), _num(value)
                ok = (a == b) if types[col] == 'real' and a is not None and b is not None else \
                    ' '.join(str(cell).lower().split()) == ' '.join(str(value).lower().split())
            else:
                a, b = _num(cell), _num(value)
                if a is None or b is None:
                    return None
                ok = a > b if op == 1 else a < b
            if not ok:
                break
        if ok:
            matched.append(row)
    agg, sel = WIKISQL_AGG[sql['agg']], sql['sel']
    values = [row[sel] for row in matched]
    if agg == '':
        return values if values else None
    if agg == 'COUNT':
        return [len(values)]
    nums = [_num(v) for v in values]
    if not nums or any(n is None for n in nums):
        return None
    result = {'MAX': max, 'MIN': min, 'SUM': sum, 'AVG': lambda xs: sum(xs) / len(xs)}[agg](nums)
    result = round(result, 6)
    return [int(result) if float(result).is_integer() else result]


def wikisql(cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('validation', cap_eval), ('test', cap_eval)):
        rows = _sample_lines(f'{BASE}/wikisql/default__{split}.jsonl', cap * 3, seed)
        n = 0
        pool = []
        for r in rows:
            table, sql = r['table'], r['sql']
            answer = wikisql_execute(table, sql)
            if answer is None or len(answer) > 8 or len({str(a).lower() for a in answer}) != len(answer):
                continue
            if any(len(str(a)) > 80 for a in answer) or len(json.dumps(table['rows'])) > 30000:
                continue
            pool.append(answer)
            agg = WIKISQL_AGG[sql['agg']] or 'select'
            unit = f'wikisql-table:{table["id"]}'
            item = f'{table["id"]}:{hashlib.sha256(r["question"].encode()).hexdigest()[:12]}'
            wrong = [w for w in ([answer[:-1]] if len(answer) > 1 else []) +
                     ([[answer[0] + 1]] if isinstance(answer[0], (int, float)) else
                      [[str(row[sql['sel']])] for row in table['rows'] if [str(row[sql['sel']])] != answer][:1])]
            if not wrong:
                continue
            out.append(make_case(
                source='wikisql', family=f'table_query_{agg.lower()}', shape=f'wikisql-{agg.lower()}-{len(sql["conds"]["column_index"])}conds',
                upstream='Salesforce/wikisql', upstream_id=item, upstream_split='validation' if split != 'train' else 'train',
                unit=unit, groups=[unit, table_content_group(table['header'], table['rows'])], root_name='answer_from_table',
                args={'question': 'string', 'table': '{ columns: string[]; rows: string[][] }'}, returns='(string | number)[]',
                instruction=pick(WIKISQL_INSTRUCTIONS, item),
                inputs={'question': r['question'], 'table': {'columns': table['header'], 'rows': table['rows']}},
                expected=answer, oracle={'level': 'normalized', 'normalization': 'answer-list'},
                level=1 + (agg != 'select') + (len(sql['conds']['column_index']) > 1), license_='BSD-3-Clause',
                license_note='HF card states unknown; original WikiSQL repo BSD-3-Clause', domain='relational',
                slice_='inline_placement', wrong=wrong, aliases=[[str(a) for a in answer]], revision='48cfb60'))
            n += 1
            if n >= cap:
                break
    return out


SQA_INSTRUCTIONS = [
    'question is the latest in a sequence of questions about table; previous_questions are the earlier ones, which '
    'it may refer back to. Answer question from the rows and return the answer cells as a list.',
    'Answer the follow-up question about table, reading previous_questions for what "those" or "which of them" refer '
    'to. Return the answering cell values as a list.',
]


def msr_sqa(cap_train=1500, cap_eval=150, seed=20261031):
    """MSR SQA: sequential questions over WikiTableQuestions tables. The split unit is the WTQ table itself
    (`wtq-table:<csv>`, as the v2 WTQ cases), so a table's SQA and WTQ items share one split."""
    out = []
    wtq_test_tables = {json.loads(line)['table_csv'] for line in open(f'{BASE}/wikitablequestions/test.jsonl')}
    for split, cap in (('train', cap_train), ('validation', cap_eval), ('test', cap_eval)):
        rows = _sample_lines(f'{BASE}/msr_sqa/default__{split}.jsonl', cap * 3, seed)
        n = 0
        for r in rows:
            answers = list(r['answer_text'])
            if not answers or len(answers) > 8 or any(len(a) > 80 for a in answers):
                continue
            if len({a.lower() for a in answers}) != len(answers):
                continue
            # SQA names WTQ's csv/203-csv/149.csv as table_csv/203_149.csv.
            prefix, number = r['table_file'].removeprefix('table_csv/').removesuffix('.csv').split('_')
            table_csv = f'csv/{prefix}-csv/{number}.csv'
            # A WTQ test table is held (dev/confirmation by the same unit bucket as its WTQ items); any other table
            # follows its WTQ train items, whatever SQA split the question came from.
            upstream_split = 'validation' if table_csv in wtq_test_tables else 'train'
            unit = f'wtq-table:{table_csv}'
            item = f'{r["id"]}:{r["annotator"]}:{r["position"]}'
            others = [[row[i]] for row in r['table_data'] for i in range(len(row)) if [row[i]] != answers][:40]
            wrong = ([answers[:-1]] if len(answers) > 1 else []) + others[len(item) % max(1, len(others)):][:1]
            out.append(make_case(
                source='msr-sqa', family='sequential_table_qa', shape=f'sqa-pos{min(r["position"], 3)}',
                upstream='microsoft/sqa', upstream_id=item, upstream_split=upstream_split, unit=unit,
                groups=[unit, f'sqa-sequence:{r["id"]}', table_content_group(r['table_header'], r['table_data'])],
                root_name='answer_followup',
                args={'question': 'string', 'previous_questions': 'string[]',
                      'table': '{ columns: string[]; rows: string[][] }'}, returns='string[]',
                instruction=pick(SQA_INSTRUCTIONS, item),
                inputs={'question': r['question'], 'previous_questions': list(r['question_and_history'][:-1]),
                        'table': {'columns': r['table_header'], 'rows': r['table_data']}},
                expected=answers, oracle={'level': 'normalized', 'normalization': 'answer-list'},
                level=1 + min(2, r['position']), license_='MS-PL', domain='relational', slice_='inline_placement',
                wrong=wrong, aliases=[list(reversed(answers)), [a.upper() for a in answers]], revision=''))
            n += 1
            if n >= cap:
                break
    return out
