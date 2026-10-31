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


def complexity(sql: str) -> int:
    s = sql.upper()
    score = s.count(' JOIN ') + 2 * max(0, s.count('SELECT') - 1) + ('GROUP BY' in s) + ('HAVING' in s) + \
        (' INTERSECT ' in s or ' UNION ' in s or ' EXCEPT ' in s) * 2
    return 0 if score == 0 else 1 if score <= 1 else 2 if score <= 3 else 3


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


def _build(source, items, db_dir_of, per_db, out_assets, upstream, revision, seed, max_mb=40):
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
        rel, digest = _asset(db_path, f'{source}-{split}-{db_id}.sqlite', out_assets)
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
            aliases = [[[str(c) if isinstance(c, (int, float)) else c for c in row] for row in rows],
                       list(reversed(rows))]
            out.append(make_case(
                source=source, family='sql_qa', shape=f'{source}-{"hint" if hint else "plain"}-L{level}',
                upstream=upstream, upstream_id=item_id, upstream_split=split, unit=f'db:{db_id}',
                groups=[f'sql-db:{source}:{db_id}'], root_name='answer_from_database', args=args, returns=ROW_TYPE,
                instruction=instruction, inputs=inputs, expected=rows,
                oracle={'level': 'normalized', 'normalization': 'result-set'}, level=level,
                license_='CC-BY-SA-4.0', services={'db': service}, domain='relational', wrong=wrong, aliases=aliases,
                neuralese_room='query planning and per-table inspection can be delegated; parent checks rows',
                revision=revision, difficulty=difficulty, neuralese=NEURALESE))
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
