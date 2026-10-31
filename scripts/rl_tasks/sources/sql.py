"""Question answering over a SQLite database (Spider, BIRD). Natlang framing: the database is a `db` service with
`schema()` and read-only `query(sql)`; the function returns the answer rows. Oracle: result-set equality with the
gold SQL's execution (a multiset of rows; numbers to 9 significant digits), BIRD's execution-accuracy rule.

Databases are copied once into the task source's `assets/sql/` and bound by SHA-256 in each case's service source;
NATLANG_RL_TASK_ASSETS may point to another copy of that directory."""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from ..common import ARCHIVE_RAW, BUILD_ASSETS, TASK_ASSETS, make_case, pick

RAW = ARCHIVE_RAW / 'agentic-20260927'
ASSET_ROOT = TASK_ASSETS
ROW_TYPE = '(string | number | null)[][]'
INSTRUCTIONS = [
    'Answer question from the database in db. Return the result as a table: one array per row, holding exactly the '
    'columns the question asks for, in the order it asks for them.',
    'question asks for data held in db. Look at the schema, query the database, and return the answer rows, each row '
    'an array of the requested column values in the order the question mentions them.',
    'Use the db service to find the rows that answer question. Return them as arrays of column values, one per row, '
    'with only the requested columns.',
]
BIRD_NOTE = '\n\nhint explains how terms in question map onto the database.'
NEURALESE = {
    'helper': 'plan_query', 'args': {'question': 'string', 'schema': 'string'}, 'returns': 'Neuralese<string>',
    'instruction': 'Read schema and write a private plan for answering question: the tables and columns involved, '
                   'how they join, the filters and aggregates, and which columns the answer must contain in what order.',
    'root_note': 'plan_query turns question and the schema text into a private query plan; use it to write and check '
                 'your query, and query the data yourself to confirm values before answering.',
}


def service_source(db_file: str, digest: str) -> str:
    return f'''import {{ DatabaseSync }} from 'node:sqlite';
import {{ createHash }} from 'node:crypto';
import {{ readFileSync }} from 'node:fs';
const FILE = {json.dumps(db_file)};
const SHA256 = {json.dumps(digest)};
function open(): any {{
  const cache = ((globalThis as any).__natlangRlSqlite ??= new Map());
  if (!cache.has(SHA256)) {{
    const root = process.env.NATLANG_RL_TASK_ASSETS ?? {json.dumps(ASSET_ROOT)};
    const path = root + '/' + FILE;
    if (createHash('sha256').update(readFileSync(path)).digest('hex') !== SHA256) throw new Error('database asset changed: ' + FILE);
    cache.set(SHA256, new DatabaseSync(path, {{ readOnly: true, enableDoubleQuotedStringLiterals: true }}));
  }}
  return cache.get(SHA256);
}}
/** The database's CREATE TABLE statements. */
export function schema(): string {{
  return open().prepare("SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL").all().map((row: any) => row.sql + ';').join('\\n');
}}
/** Run one read-only SQL query (SQLite dialect) and return its rows, each an array of column values. */
export function query(sql: string): (string | number | null)[][] {{
  const statement = open().prepare(sql);
  if (!statement.columns().length) throw new Error('only SELECT queries are allowed');
  statement.setReturnArrays(true);
  return statement.all().map((row: any) => row.map((value: any) =>
    typeof value === 'bigint' ? Number(value) : value instanceof Uint8Array ? '<blob>' : value));
}}
'''


def execute(path: Path, sql: str, seconds=10.0):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    con.text_factory = lambda b: b.decode('utf-8', 'replace')
    deadline = time.monotonic() + seconds
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20000)
    try:
        return [list(row) for row in con.execute(sql).fetchall()]
    finally:
        con.close()


def sql_profile(sql: str) -> dict:
    """Small lexical feature profile for curriculum routing, not a SQL parser or correctness proof."""
    # Ignore quoted literals and comments so words like "join" in a text constant do not change the profile.
    s = re.sub(r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", ' ', sql,
               flags=re.S).upper()
    joins = len(re.findall(r'\bJOIN\b', s))
    selects = len(re.findall(r'\bSELECT\b', s))
    group_by = bool(re.search(r'\bGROUP\s+BY\b', s))
    having = bool(re.search(r'\bHAVING\b', s))
    set_operation = bool(re.search(r'\b(?:INTERSECT|UNION|EXCEPT)\b', s))
    order_by = bool(re.search(r'\bORDER\s+BY\b', s))
    where_predicates = len(re.findall(r'\b(?:WHERE|AND|OR)\b', s))
    aggregate = bool(re.search(r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(', s))
    nested_selects = max(0, selects - 1)
    # Scores distinct operations rather than merely SELECT count. Hard eligibility below requires composition.
    score = 2 * joins + 3 * nested_selects + 2 * (group_by and having) + 2 * set_operation + \
        (aggregate and joins > 0) + (order_by and where_predicates > 0 and joins > 0)
    return {'joins': joins, 'nested_selects': nested_selects, 'group_by': group_by, 'having': having,
            'set_operation': set_operation, 'order_by': order_by, 'where_predicates': where_predicates,
            'aggregate': aggregate, 'score': int(score),
            'compositional': bool(joins >= 2 or (nested_selects >= 1 and (joins >= 1 or group_by)) or
                                  (group_by and having) or (set_operation and (joins or nested_selects)) or
                                  (aggregate and joins >= 1 and where_predicates >= 2))}


def complexity(sql: str) -> int:
    p = sql_profile(sql)
    score = p['score'] + int(p['order_by'])
    return 0 if score == 0 else 1 if score <= 2 else 2 if score <= 5 else 3


def operation_signature(sql: str) -> tuple:
    """Bounded family signature used to spread hard examples across distinct SQL operations."""
    p = sql_profile(sql)
    return (min(p['joins'], 3), min(p['nested_selects'], 2), p['group_by'], p['having'],
            p['set_operation'], p['order_by'], p['aggregate'], min(p['where_predicates'], 3))


def diverse_query_order(items, rng):
    """Round-robin deterministic query-signature buckets; query text remains the correctness oracle."""
    buckets = defaultdict(list)
    for item in items:
        buckets[operation_signature(item[3])].append(item)
    signatures = sorted(buckets)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    rng.shuffle(signatures)
    ordered = []
    while signatures:
        next_signatures = []
        for signature in signatures:
            bucket = buckets[signature]
            if bucket:
                ordered.append(bucket.pop())
            if bucket:
                next_signatures.append(signature)
        signatures = next_signatures
    return ordered


def usable(rows) -> bool:
    if not rows or len(rows) > 30:
        return False
    cells = [c for row in rows for c in row]
    if all(c is None for c in cells):
        return False
    return all(isinstance(c, (int, float, str)) or c is None for c in cells) and \
        all(not isinstance(c, str) or len(c) <= 300 for c in cells)


def _asset(db_path: Path, name: str, out_assets: Path):
    target = out_assets / 'sql' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(db_path, target)
    return f'sql/{name}', hashlib.sha256(target.read_bytes()).hexdigest()


def _build(source, items, db_dir_of, per_db, out_assets, upstream, revision, seed, max_mb=40,
           compositional_only=False, group_source=None):
    """items: (upstream_split, db_id, question, sql, hint, difficulty, item_id); split by database."""
    by_db = defaultdict(list)
    for item in items:
        by_db[(item[0], item[1])].append(item)
    out = []
    rng = random.Random(seed)
    for (split, db_id), members in sorted(by_db.items()):
        db_path = db_dir_of(split, db_id)
        if not db_path.exists() or db_path.stat().st_size > max_mb * 2**20:
            continue
        if compositional_only:
            members = [item for item in members if sql_profile(item[3])['compositional']]
            members = diverse_query_order(members, rng)
        else:
            rng.shuffle(members)
        results = []
        for item in members:
            sql = item[3]
            if re.search(r'\bLIMIT\b', sql, re.I) and not re.search(r'\bORDER\s+BY\b', sql, re.I):
                continue
            try:
                rows = execute(db_path, sql)
            except Exception:
                continue
            if usable(rows):
                results.append((item, rows))
            if len(results) >= per_db:
                break
        if not results:
            continue
        rel, digest = _asset(db_path, f'{group_source or source}-{split}-{db_id}.sqlite', out_assets)
        service = service_source(rel, digest)
        golds = [rows for _, rows in results]
        for index, ((_, _, question, sql, hint, difficulty, item_id), rows) in enumerate(results):
            others = [g for g in golds if json.dumps(g) != json.dumps(rows)][:1]
            wrong = others + ([rows[:-1]] if len(rows) > 1 else [[[r[0] for r in rows][0], 'extra']] if len(rows[0]) == 1 else [])
            level = complexity(sql) if difficulty is None else {'simple': 1, 'moderate': 2, 'challenging': 3}.get(difficulty, 2)
            args = {'question': 'string'}
            inputs = {'question': question}
            instruction = pick(INSTRUCTIONS, f'{db_id}|{item_id}')
            if hint:
                args['hint'] = 'string'
                inputs['hint'] = hint
                instruction += BIRD_NOTE
            ordered = sql_profile(sql)['order_by']
            aliases = [[[str(c) if isinstance(c, (int, float)) else c for c in row] for row in rows]]
            if not ordered:
                aliases.append(list(reversed(rows)))
            out.append(make_case(
                source=source, family='sql_qa', shape=f'{source}-{"hint" if hint else "plain"}-L{level}',
                upstream=upstream, upstream_id=item_id, upstream_split=split, unit=f'db:{db_id}',
                groups=[f'sql-db:{group_source or source}:{db_id}'], root_name='answer_from_database', args=args, returns=ROW_TYPE,
                instruction=instruction, inputs=inputs, expected=rows,
                oracle={'level': 'normalized', 'normalization': 'result-set', **({'ordered': True} if ordered else {})}, level=level,
                license_='CC-BY-SA-4.0', services={'db': service}, domain='relational', wrong=wrong, aliases=aliases,
                neuralese_room='query planning and per-table inspection can be delegated; parent checks rows',
                revision=revision, difficulty=difficulty, neuralese=NEURALESE))
            out[-1]['curriculum']['external']['sql_profile'] = sql_profile(sql)
            out[-1]['curriculum']['external']['gold_sql'] = sql
    return out


def spider(out_assets=BUILD_ASSETS, per_db=8, seed=20261031):
    base = RAW / 'spider/official/spider_data'
    train = json.load(open(base / 'train_spider.json')) + json.load(open(base / 'train_others.json'))
    dev = json.load(open(base / 'dev.json'))
    test = json.load(open(base / 'test.json'))
    seen_dbs = {r['db_id'] for r in train} | {r['db_id'] for r in dev}
    items = [('train', r['db_id'], r['question'], r['query'], '', None, f'train-{i}') for i, r in enumerate(train)]
    items += [('dev', r['db_id'], r['question'], r['query'], '', None, f'dev-{i}') for i, r in enumerate(dev)]
    items += [('test', r['db_id'], r['question'], r['query'], '', None, f'test-{i}') for i, r in enumerate(test)
              if r['db_id'] not in seen_dbs]

    def db_dir(split, db_id):
        return (base / ('test_database' if split == 'test' else 'database') / db_id / f'{db_id}.sqlite')
    return _build('spider', items, db_dir, per_db, Path(out_assets), 'spider', 'spider_data.zip-00636695', seed)


def bird(out_assets=BUILD_ASSETS, per_db=40, seed=20261031):
    train = json.load(open(RAW / 'bird/train/train.json'))
    dev = json.load(open(RAW / 'bird/dev_20240627/dev.json'))
    items = [('train', r['db_id'], r['question'], r['SQL'], r.get('evidence') or '', None, f'train-{i}')
             for i, r in enumerate(train)]
    items += [('dev', r['db_id'], r['question'], r['SQL'], r.get('evidence') or '', r.get('difficulty'),
               f'dev-{r.get("question_id", i)}') for i, r in enumerate(dev)]

    def db_dir(split, db_id):
        return RAW / ('bird/train/train_databases' if split == 'train' else 'bird/dev_20240627/dev_databases') / \
            db_id / f'{db_id}.sqlite'
    return _build('bird', items, db_dir, per_db, Path(out_assets), 'bird-bench', 'train.zip+dev_20240627', seed)


def _focused(source, items, db_dir, per_db, out_assets, upstream, revision, seed, group_source):
    """Build a new, disjoint-ID hard SQL successor with multi-operation queries only."""
    # A database is the related-world unit. If a DB has any held rows, keep its
    # whole DB group held; never split train queries from dev queries by row.
    held_databases = {item[1] for item in items if item[0] != 'train'}
    items = [item for item in items if item[0] != 'train' or item[1] not in held_databases]
    return _build(source, items, db_dir, per_db, Path(out_assets), upstream,
                  revision + '+compositional-sql-v1', seed, compositional_only=True,
                  group_source=group_source)


def spider_hard(out_assets=BUILD_ASSETS, per_db=12, seed=20261031):
    base = RAW / 'spider/official/spider_data'
    train = json.load(open(base / 'train_spider.json')) + json.load(open(base / 'train_others.json'))
    dev = json.load(open(base / 'dev.json'))
    test = json.load(open(base / 'test.json'))
    seen_dbs = {r['db_id'] for r in train} | {r['db_id'] for r in dev}
    items = [('train', r['db_id'], r['question'], r['query'], '', None, f'train-{i}') for i, r in enumerate(train)]
    items += [('dev', r['db_id'], r['question'], r['query'], '', None, f'dev-{i}') for i, r in enumerate(dev)]
    items += [('test', r['db_id'], r['question'], r['query'], '', None, f'test-{i}') for i, r in enumerate(test)
              if r['db_id'] not in seen_dbs]
    def db_dir(split, db_id):
        return (base / ('test_database' if split == 'test' else 'database') / db_id / f'{db_id}.sqlite')
    return _focused('spider-hard', items, db_dir, per_db, out_assets, 'spider',
                    'spider_data.zip-00636695', seed, 'spider')


def bird_hard(out_assets=BUILD_ASSETS, per_db=24, seed=20261031):
    train = json.load(open(RAW / 'bird/train/train.json'))
    dev = json.load(open(RAW / 'bird/dev_20240627/dev.json'))
    items = [('train', r['db_id'], r['question'], r['SQL'], r.get('evidence') or '', None, f'train-{i}')
             for i, r in enumerate(train)]
    items += [('dev', r['db_id'], r['question'], r['SQL'], r.get('evidence') or '', r.get('difficulty'),
               f'dev-{r.get("question_id", i)}') for i, r in enumerate(dev)]
    def db_dir(split, db_id):
        return RAW / ('bird/train/train_databases' if split == 'train' else 'bird/dev_20240627/dev_databases') / \
            db_id / f'{db_id}.sqlite'
    return _focused('bird-hard', items, db_dir, per_db, out_assets, 'bird-bench',
                    'train.zip+dev_20240627', seed, 'bird')
