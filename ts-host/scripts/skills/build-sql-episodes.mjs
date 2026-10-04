#!/usr/bin/env node
/** Text-to-SQL skill episodes from Spider or BIRD, scored by host-only result-set F1. No model calls.
 *
 * BIRD (`--source bird`) is harder: larger databases and questions that come with an evidence hint, passed to the
 * target as a third argument. Its train and dev databases live in different folders, so gold paths are relative to
 * the BIRD root, which is then the collection's database root.
 *
 * Each episode is one database (the environment). Support and query are different questions on it,
 * with distinct normalized gold SQL; transfer questions come from another database of the same split.
 * Train episodes use Spider train databases; validation/test episodes use the disjoint dev databases. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { runReadOnlyQuery } from '../../dist/skills/graded.js';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
const source = arg('--source') ?? 'spider';
if (!['spider', 'bird'].includes(source)) throw Error('--source is spider or bird');
const spider = resolve(arg('--spider') ?? '/mnt/external/sdkb-archive/raw/agentic-20260927/spider/official/spider_data');
const bird = resolve(arg('--bird') ?? '/mnt/external/sdkb-archive/raw/agentic-20260927/bird');
const out = arg('--out');
const perDb = Number(arg('--per-db') ?? 1), maxDbs = Number(arg('--max-dbs') ?? Infinity);
if (!out || !Number.isSafeInteger(perDb) || perDb < 1)
  throw Error('Usage: build-sql-episodes.mjs --out DIR [--source spider|bird] [--spider DIR] [--bird DIR] [--per-db N] [--max-dbs N]');
const databaseRoot = source === 'bird' ? bird : join(spider, 'database');
const LICENSE = source === 'bird' ? 'CC-BY-SA-4.0 (BIRD)' : 'CC-BY-SA-4.0 (Spider)';
/** Database file of `db`, relative to the database root (BIRD keeps train and dev databases apart). */
const dbPaths = new Map();
const dbPath = db => dbPaths.get(db) ?? `${db}/${db}.sqlite`;
const familyOf = db => source === 'spider' ? `text-to-sql:${db}` : `text-to-sql:${source}:${db}`;
const normalizeSql = sql => sql.toLowerCase().replace(/\s+/g, ' ').replace(/\s*([(),=<>])\s*/g, '$1').replace(/;\s*$/, '').trim();

function schemaText(db) {
  const handle = new DatabaseSync(join(databaseRoot, dbPath(db)), { readOnly: true });
  try {
    return handle.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name").all()
      .map(row => String(row.sql).trim() + ';').join('\n');
  } finally { handle.close(); }
}

/** Usable questions: gold executes, returns 1..200 rows, and its normalized SQL is unique within the database. */
function usable(rows) {
  const byDb = new Map();
  for (const row of rows) {
    const list = byDb.get(row.db_id) ?? []; list.push(row); byDb.set(row.db_id, list);
  }
  const result = new Map();
  for (const [db, list] of [...byDb].sort(([a], [b]) => a < b ? -1 : 1)) {
    const seen = new Set(), keep = [];
    // Candidates in a deterministic order; BIRD stops once a database has enough usable questions (its databases are
    // large). Spider keeps every usable question, so its episodes, including transfer tails, stay as built before.
    list.sort((a, b) => digest(normalizeSql(a.query)) < digest(normalizeSql(b.query)) ? -1 : 1);
    for (const row of list) {
      if (source === 'bird' && keep.length >= 16 * perDb) break;
      const norm = normalizeSql(row.query);
      if (seen.has(norm)) continue;
      seen.add(norm);
      // The gold query must run as the scorer runs it (its row limit), within half the scorer's timeout, so a
      // loaded machine cannot turn a gold answer into an unscoreable case (the episode gate's gold check).
      const executed = runReadOnlyQuery(join(databaseRoot, dbPath(db)), row.query, { timeoutMs: 2500 });
      if (executed.error || !executed.rows?.length || executed.rows.length > 200) continue;
      keep.push({ ...row, norm });
    }
    // Deterministic order independent of the source file's order.
    keep.sort((a, b) => digest(a.norm) < digest(b.norm) ? -1 : 1);
    result.set(db, keep);
  }
  return result;
}

const target = () => source === 'bird' ? { kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
  source: { schema: 'natlang.skill-sql-target/1', id: 'bird-text-to-sql-v1' },
  files: { 'solve.nl': '---\nargs: { question: string, evidence: string, schema: string }\nreturns: string\n---\nWrite one SQLite SELECT query that answers the question over the given schema. The evidence explains terms and formulas the question relies on. Return only the SQL. The host executes it read-only and compares its result rows with the reference answer.\n' } }
  : { kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
  source: { schema: 'natlang.skill-sql-target/1', id: 'spider-text-to-sql-v1' },
  files: { 'solve.nl': '---\nargs: { question: string, schema: string }\nreturns: string\n---\nWrite one SQLite SELECT query that answers the question over the given schema. Return only the SQL. The host executes it read-only and compares its result rows with the reference answer.\n' } };

function cases(db, rows, schema) {
  return rows.map(row => ({ id: `case-${digest(db + '\n' + row.norm).slice(0, 20)}`, group: `g-${digest(source + ':' + db + '\n' + row.norm).slice(0, 24)}`,
    args: source === 'bird' ? [row.question, row.evidence ?? '', schema] : [row.question, schema],
    expected: { kind: 'sql-gold', db: dbPath(db), sql: row.query } }));
}

function episodes(byDb, split) {
  const dbs = [...byDb.keys()].filter(db => byDb.get(db).length >= 8 * perDb).slice(0, maxDbs);
  const schemas = new Map(dbs.map(db => [db, schemaText(db)]));
  const built = [];
  dbs.forEach((db, index) => {
    const other = dbs[(index + 1) % dbs.length];
    for (let n = 0; n < perDb; n++) {
      const pool = byDb.get(db).slice(8 * n, 8 * n + 8);
      const support = cases(db, pool.slice(0, 4), schemas.get(db)), query = cases(db, pool.slice(4, 8), schemas.get(db));
      // Transfer draws from the other database's tail, which that database's own episodes do not use.
      const otherRows = byDb.get(other), tail = otherRows.slice(otherRows.length - 4 * (n + 1), otherRows.length - 4 * n);
      const transfer = other !== db && otherRows.length >= 8 * perDb + 4 * perDb ? cases(other, tail, schemas.get(other)) : null;
      const groups = [...support, ...query, ...(transfer ?? [])].map(c => c.group).sort();
      built.push({ version: 'natlang.skill-episode/1', id: `skill-sql-${source}-${db}-${n}`, family: familyOf(db), split,
        source_groups: [`group-commitment:sha256:${digest(groups)}`], license: LICENSE,
        target: target(), library: { kind: 'empty', skills: {} },
        support: { cases: support }, query: { cases: query },
        ...(transfer ? { transfer: { family: familyOf(other), target: target(), cases: transfer } } : {}),
        operations: ['create', 'revise', 'select', 'test'], limits: { maxSteps: 6 },
        provenance: { generator: 'natlang.skill-sql-episodes/1', source, database: db, transfer_database: transfer ? other : null,
          metric: { schema: 'natlang.skill-graded/1', kind: 'sql-result-f1' },
          ...(transfer ? { transfer_metric: { schema: 'natlang.skill-graded/1', kind: 'sql-result-f1' } } : {}) } });
    }
  });
  return built;
}

async function loadBird(file, folder) {
  const rows = JSON.parse(await readFile(join(bird, file), 'utf8'));
  for (const row of rows) dbPaths.set(row.db_id, `${folder}/${row.db_id}/${row.db_id}.sqlite`);
  return rows.map(row => ({ ...row, query: row.SQL }));
}
const train = source === 'bird' ? await loadBird('train/train.json', 'train/train_databases')
  : JSON.parse(await readFile(join(spider, 'train_spider.json'), 'utf8'));
const dev = source === 'bird' ? await loadBird('dev_20240627/dev.json', 'dev_20240627/dev_databases')
  : JSON.parse(await readFile(join(spider, 'dev.json'), 'utf8'));
const trainEpisodes = episodes(usable(train), 'train');
// Dev databases are disjoint from train databases; alternate them into validation and test episodes.
const heldout = episodes(usable(dev), 'heldout').map((row, i) => ({ ...row, split: i % 2 ? 'test' : 'validation' }));
const rows = [...trainEpisodes, ...heldout];
const all = rows.flatMap(row => [...row.support.cases, ...row.query.cases, ...(row.transfer?.cases ?? [])]);
await mkdir(resolve(out), { recursive: true });
const body = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
await writeFile(join(resolve(out), 'sql-episodes.jsonl'), body, { flag: 'wx' });
const manifest = { schema: 'natlang.skill-sql-episodes/1', episodes: rows.length,
  splits: Object.fromEntries(['train', 'validation', 'test'].map(s => [s, rows.filter(r => r.split === s).length])),
  source, cases: all.length, database_root: databaseRoot, per_db: perDb, license: LICENSE, model_calls: 0,
  sha256: digest(body), note: 'Collect with --database-root set to database_root; scores execute SQL read-only on the host.' };
await writeFile(join(resolve(out), 'sql-episodes.manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(manifest, null, 2));
// The packet gates itself (scripts/skills/audit-episodes.mjs): gold queries must reach their best score.
const gate = spawnSync(process.execPath, [fileURLToPath(new URL('./audit-episodes.mjs', import.meta.url)),
  join(resolve(out), 'sql-episodes.jsonl'), '--database-root', databaseRoot, '--out', join(resolve(out), 'sql-episodes.audit.json')],
  { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const verdict = JSON.parse(gate.stdout || '{}');
console.log(JSON.stringify({ gate: gate.status === 0 ? 'passed' : 'failed', errors: verdict.error_count, warnings: verdict.warning_count }));
if (gate.status !== 0) process.exitCode = 1;
