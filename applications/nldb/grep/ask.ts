/**
 * Questions in words over a repository and its metadata tables (WS6 §3 A8, built on nldb's SQLite engine).
 *
 * The files become a `files(path, sha256, content)` table and each metadata table (CSV rows) a table of its own, in one
 * SQLite database; nldb's `SqliteDatabase.ask` then compiles the question (`sqlite/translate.nl`: exact SQL, with every
 * condition on meaning written `meets(column, 'criterion')`) and judges each distinct stored value once
 * (`sqlite/judge.nl`), keeping the verdicts in the database: asking again, or asking another question with the same
 * criterion, judges nothing new. Exact joins, filters, grouping and ordering are SQLite's.
 *
 * `runQuery` (query.ts) stays the path for a plan the caller writes.
 */
import { DatabaseSync } from 'node:sqlite';
import { SqliteDatabase, type Run } from '../index.js';
import type { Outcome } from '../types.js';
import type { QueryDataset, TableValue } from './types.js';

const quote = (name: string) => `"${name.replace(/"/g, '""')}"`;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

function columnType(values: readonly TableValue[]): string {
  const present = values.filter(value => value !== null && value !== undefined);
  if (present.length && present.every(value => typeof value === 'number' && Number.isInteger(value))) return 'INTEGER';
  if (present.length && present.every(value => typeof value === 'number')) return 'REAL';
  return 'TEXT';
}

/**
 * Write `dataset` into the SQLite database at `path`: the files (replacing a file's row when its content changed) and
 * each table (recreated from its rows). The `_nl_semantic` verdicts nldb keeps there survive, so they are reused.
 */
export function loadDataset(path: string, dataset: QueryDataset): void {
  const db = new DatabaseSync(path);
  try {
    db.exec('BEGIN');
    db.exec('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, content TEXT NOT NULL)');
    db.exec('DELETE FROM files');
    const insertFile = db.prepare('INSERT INTO files (path, sha256, content) VALUES (?, ?, ?)');
    for (const file of dataset.files) insertFile.run(file.path, file.sha256, String(file.content));
    for (const [table, rows] of Object.entries(dataset.tables ?? {})) {
      if (!NAME.test(table) || table === 'files' || table.startsWith('_nl')) throw new Error(`table name ${JSON.stringify(table)} is not a plain identifier other than files`);
      const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
      if (!columns.length) continue;
      const bad = columns.find(column => !NAME.test(column));
      if (bad) throw new Error(`column ${JSON.stringify(bad)} of ${table} is not a plain identifier`);
      db.exec(`DROP TABLE IF EXISTS ${quote(table)}`);
      db.exec(`CREATE TABLE ${quote(table)} (${columns.map(column => `${quote(column)} ${columnType(rows.map(row => row[column] ?? null))}`).join(', ')})`);
      const insert = db.prepare(`INSERT INTO ${quote(table)} (${columns.map(quote).join(', ')}) VALUES (${columns.map(() => '?').join(', ')})`);
      for (const row of rows) insert.run(...columns.map(column => {
        const value = row[column];
        return value === undefined ? null : typeof value === 'boolean' ? (value ? 1 : 0) : value;
      }));
    }
    db.exec('COMMIT');
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  } finally { db.close(); }
}

/**
 * Answer `question` over `dataset` with nldb's SQLite engine, the database kept at `path` (a file the caller keeps to
 * reuse semantic verdicts across runs). `run` runs a natural-language call (`fn => runtime.run(fn)`).
 */
export async function askRepository(question: string, dataset: QueryDataset, options: { path: string; run: Run; today?: () => string }): Promise<Outcome> {
  loadDataset(options.path, dataset);
  const database = new SqliteDatabase(options.path, options.run, options.today);
  try {
    const outcome = await database.ask(question);
    if (outcome.kind === 'change' || outcome.kind === 'schema')
      return { kind: outcome.kind, error: 'a question over a repository reads it; it does not change the files or their tables' };
    return outcome;
  } finally { database.close(); }
}
