/** Host-only graded scores for skill episodes drawn from real data (no model judge). */
import { spawnSync } from 'node:child_process';
import { isAbsolute, join, normalize } from 'node:path';

export const GRADED_KINDS = ['sql-result-f1'] as const;
export type GradedKind = typeof GRADED_KINDS[number];
export type GradedMetric = { schema: 'natlang.skill-graded/1'; kind: GradedKind; database_root?: string };
export type GradedScore = { quality: number; gates: Record<string, boolean>; detail?: Record<string, unknown> };

const invalid = (gate: string, detail?: Record<string, unknown>): GradedScore => ({ quality: 0, gates: { [gate]: false }, detail });

/** Runs read-only queries in a child process so that a runaway query cannot stall the host. */
const RUNNER = `
const { DatabaseSync } = require('node:sqlite');
const [path, sql, limit] = JSON.parse(process.argv[1]);
try {
  const db = new DatabaseSync(path, { readOnly: true });
  const statement = db.prepare(sql);
  const rows = []; for (const row of statement.iterate()) { rows.push(Object.values(row)); if (rows.length > limit) break; }
  process.stdout.write(JSON.stringify({ rows }));
} catch (error) { process.stdout.write(JSON.stringify({ error: String(error && error.message || error) })); }
`;

export function runReadOnlyQuery(path: string, sql: string, options: { timeoutMs?: number; limit?: number } = {}):
  { rows?: unknown[][]; error?: string } {
  const result = spawnSync(process.execPath, ['--no-warnings', '-e', RUNNER, JSON.stringify([path, sql, options.limit ?? 5000])],
    { timeout: options.timeoutMs ?? 5000, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) return { error: result.error ? String(result.error.message) : `exit ${result.status}` };
  try { return JSON.parse(result.stdout); } catch { return { error: 'unreadable query output' }; }
}

/** Rows compare as multisets; integral floats and integers are the same value. */
function key(row: unknown[]): string {
  return JSON.stringify(row.map(cell => typeof cell === 'number' && Number.isInteger(cell) ? cell
    : typeof cell === 'number' ? Number(cell.toPrecision(12)) : cell));
}

export function multisetF1(predicted: unknown[][], gold: unknown[][]): { f1: number; precision: number; recall: number } {
  if (!predicted.length && !gold.length) return { f1: 1, precision: 1, recall: 1 };
  const counts = new Map<string, number>();
  for (const row of gold) counts.set(key(row), (counts.get(key(row)) ?? 0) + 1);
  let hit = 0;
  for (const row of predicted) { const k = key(row), n = counts.get(k) ?? 0; if (n > 0) { hit++; counts.set(k, n - 1); } }
  const precision = predicted.length ? hit / predicted.length : 0, recall = gold.length ? hit / gold.length : 0;
  return { f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0, precision, recall };
}

function extractSql(value: unknown): string | undefined {
  let text = typeof value === 'string' ? value : value && typeof value === 'object' && typeof (value as any).sql === 'string' ? (value as any).sql : undefined;
  if (text === undefined) return undefined;
  const fenced = /```(?:sql)?\s*([\s\S]*?)```/i.exec(text);
  if (fenced) text = fenced[1]!;
  return text.trim().replace(/;\s*$/, '');
}

/** Result-set F1 of a returned SQL query against the gold query, both executed on the episode database. */
export function scoreGraded(metric: GradedMetric, value: unknown, expected: unknown): GradedScore {
  if (metric.kind !== 'sql-result-f1') return invalid('graded_kind');
  const gold = expected as { kind?: string; db?: string; sql?: string } | null;
  if (!gold || gold.kind !== 'sql-gold' || typeof gold.db !== 'string' || typeof gold.sql !== 'string') return invalid('valid_reference');
  if (!metric.database_root || isAbsolute(gold.db) || normalize(gold.db).startsWith('..')) return invalid('database_path');
  const path = join(metric.database_root, gold.db);
  const sql = extractSql(value);
  if (!sql) return invalid('returned_sql');
  if (/;\s*\S/.test(sql)) return invalid('single_statement');
  const reference = runReadOnlyQuery(path, gold.sql);
  if (reference.error || !reference.rows) return invalid('reference_executes', { error: reference.error });
  const predicted = runReadOnlyQuery(path, sql);
  if (predicted.error || !predicted.rows) return invalid('executes', { error: predicted.error });
  const { f1, precision, recall } = multisetF1(predicted.rows, reference.rows);
  return { quality: f1, gates: { executes: true, exact_result: f1 === 1 }, detail: { precision, recall, rows: predicted.rows.length, gold_rows: reference.rows.length } };
}
