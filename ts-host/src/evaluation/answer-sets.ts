/**
 * Generic answer comparators for dataset-backed RL tasks (plans/neuralese/BROAD_RL_TASKS_20261031.md). They are
 * selected by `oracle: { level: 'normalized', normalization: <name> }` and implemented once here for every source:
 *
 * - `result-set`: rows of a query result (SQL execution match). A multiset of rows unless `ordered`; a value is a
 *   number when it reads as one (so 3, "3" and 3.0 agree), numbers compare to 9 significant digits, text is trimmed.
 *   A scalar answer is one row of one cell; a flat list is one cell per row; an object row lists its values in order.
 * - `numeric-tolerance`: one number within a relative (`tolerance`, default 1e-4) or absolute (`abs_tolerance`)
 *   distance; `percent_equivalent` also accepts the value scaled by 100 either way (0.25 versus 25%).
 * - `answer-list`: an unordered list of short answers (table QA); a scalar answers a one-item list.
 * - `qa-alias-map`: a record of free-text answers by key; each key's answer must reach token F1 `threshold`
 *   (default 0.8, SQuAD normalization, numbers exact) against one of that key's gold aliases.
 */
import { canonical } from '../oracle-kit/json-record.js';

export const ANSWER_SET_NORMALIZATIONS = ['result-set', 'numeric-tolerance', 'answer-list', 'qa-alias-map'] as const;
export type AnswerSetNormalization = typeof ANSWER_SET_NORMALIZATIONS[number];

const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

/** A number from a number or a plain numeric string (thousands separators, currency and percent signs dropped). */
export function readNumber(value: unknown, loose = false): number | undefined {
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined;
  if (typeof value !== 'string') return undefined;
  let text = value.normalize('NFKC').trim().replace(/(?<=\d),(?=\d{3}(?:\D|$))/g, '');
  if (loose) text = text.replace(/^\$\s*/, '').replace(/^\((.*)\)$/, '-$1').replace(/\s*(?:%|percent)$/i, '').trim();
  return NUMBER.test(text) ? Number(text) : undefined;
}

function cell(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'boolean') return `n:${value ? 1 : 0}`;
  const number = readNumber(value);
  if (number !== undefined) return `n:${Number(number.toPrecision(9))}`;
  if (typeof value === 'string') return `s:${value.normalize('NFKC').trim()}`;
  return `j:${canonical(value)}`;
}

function rows(value: unknown): string[][] | undefined {
  if (!Array.isArray(value)) return value !== null && typeof value === 'object' ? undefined : [[cell(value)]];
  return value.map(row => Array.isArray(row) ? row.map(cell) :
    row !== null && typeof row === 'object' ? Object.values(row as Record<string, unknown>).map(cell) : [cell(row)]);
}

export function resultSetEqual(actual: unknown, expected: unknown, ordered = false): boolean {
  const left = rows(actual), right = rows(expected);
  if (!left || !right || left.length !== right.length) return false;
  const key = (table: string[][]) => { const lines = table.map(row => JSON.stringify(row)); return ordered ? lines : lines.sort(); };
  return canonical(key(left)) === canonical(key(right));
}

export function numericClose(actual: unknown, expected: unknown, spec: { tolerance?: number; abs_tolerance?: number;
  percent_equivalent?: boolean } = {}): boolean {
  const got = readNumber(actual, true), want = readNumber(expected, true);
  if (got === undefined || want === undefined) return false;
  const relative = spec.tolerance ?? 1e-4, absolute = spec.abs_tolerance ?? 0;
  const close = (a: number, b: number) => Math.abs(a - b) <= Math.max(absolute, relative * Math.max(Math.abs(a), Math.abs(b)));
  return close(got, want) || !!spec.percent_equivalent && (close(got, want * 100) || close(got * 100, want));
}

function shortAnswer(value: unknown): string {
  const number = readNumber(value);
  if (number !== undefined) return `n:${Number(number.toPrecision(9))}`;
  return String(value ?? '').normalize('NFKC').toLocaleLowerCase('en-US').replace(/[\p{P}\p{S}]/gu, ' ')
    .split(/\s+/u).filter(token => token && !['a', 'an', 'the'].includes(token)).join(' ');
}

export function answerListEqual(actual: unknown, expected: unknown): boolean {
  const want = (Array.isArray(expected) ? expected : [expected]).map(shortAnswer).sort();
  const got = (Array.isArray(actual) ? actual : [actual]).map(shortAnswer).sort();
  return canonical(got) === canonical(want);
}

export function qaAliasMapScore(actual: unknown, expected: unknown, f1: (actual: unknown, gold: unknown) => number,
    threshold = 0.8): { accepted: boolean; failed: string[] } {
  let record = actual;
  if (typeof record === 'string') { try { record = JSON.parse(record); } catch { return { accepted: false, failed: ['<not a record>'] }; } }
  if (!record || typeof record !== 'object' || Array.isArray(record) || !expected || typeof expected !== 'object')
    return { accepted: false, failed: ['<not a record>'] };
  const gold = expected as Record<string, unknown>, got = record as Record<string, unknown>;
  const failed = Object.keys(got).filter(key => !Object.hasOwn(gold, key));
  for (const [key, aliases] of Object.entries(gold)) {
    const options = Array.isArray(aliases) ? aliases : [aliases];
    const answer = got[key];
    if (typeof answer !== 'string' && typeof answer !== 'number') { failed.push(key); continue; }
    if (Math.max(0, ...options.map(alias => f1(String(answer), String(alias)))) < threshold) failed.push(key);
  }
  return { accepted: failed.length === 0, failed };
}
