/** Answer checks used by generated and dataset-backed teacher cases. */
export type OracleLevel = 'exact' | 'normalized' | 'span' | 'agreement' | 'judged';
export const ORACLE_LEVELS: readonly OracleLevel[] = ['exact', 'normalized', 'span', 'agreement', 'judged'];
export type OracleSpec = OracleLevel | { level: OracleLevel; alternates?: unknown[];
  threshold?: number; rubric?: string; [key: string]: unknown };
export type OracleVerdict = { accepted: boolean; level: OracleLevel; score?: number; verdict?: string };

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
function normalizeText(value: unknown): string {
  const text = String(value ?? '').normalize('NFKC').trim().toLocaleLowerCase('en-US').replace(/\s+/g, ' ');
  const number = text.replace(/(?<=\d),(?=\d{3}(?:\D|$))/g, '');
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(number)) return `number:${Number(number)}`;
  const date = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(text);
  if (date) return `date:${date[3]}-${date[1]!.padStart(2, '0')}-${date[2]!.padStart(2, '0')}`;
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return `date:${text}`;
  return text;
}
function normalized(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(normalized).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${normalized(child)}`).join(',')}}`;
  return normalizeText(value);
}
function words(value: unknown): string[] {
  return String(Array.isArray(value) ? value.join(' ') : value ?? '').normalize('NFKC')
    .toLocaleLowerCase('en-US').match(/[\p{L}\p{N}]+/gu) ?? [];
}
/** Multiset token F1; repeated words count only when they occur on both sides. */
export function spanF1(actual: unknown, expected: unknown): number {
  const left = words(actual), right = words(expected);
  if (!left.length || !right.length) return left.length === right.length ? 1 : 0;
  const counts = new Map<string, number>();
  for (const word of right) counts.set(word, (counts.get(word) ?? 0) + 1);
  let shared = 0;
  for (const word of left) if ((counts.get(word) ?? 0) > 0) { shared++; counts.set(word, counts.get(word)! - 1); }
  return 2 * shared / (left.length + right.length);
}

/**
 * How much of a many-part answer matches, from 0 to 1: for answers made of many judgments whose gold is itself noisy
 * (dataset labels). A record of numbers (counts by label) scores 1 - L1 distance / both totals, an array the share of
 * equal positions, a number 1 - its relative error; anything else is all or nothing.
 */
export function agreement(actual: unknown, expected: unknown): number {
  const numbers = (value: unknown): value is Record<string, number> => !!value && typeof value === 'object' &&
    !Array.isArray(value) && Object.values(value).every(item => typeof item === 'number');
  if (numbers(actual) && numbers(expected)) {
    const keys = new Set([...Object.keys(actual), ...Object.keys(expected)]);
    const total = [...keys].reduce((sum, key) => sum + Math.abs(actual[key] ?? 0) + Math.abs(expected[key] ?? 0), 0);
    const distance = [...keys].reduce((sum, key) => sum + Math.abs((actual[key] ?? 0) - (expected[key] ?? 0)), 0);
    return total ? 1 - distance / total : 1;
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    const length = Math.max(actual.length, expected.length);
    return length ? expected.filter((item, index) => canonical(item) === canonical(actual[index])).length / length : 1;
  }
  if (typeof actual === 'number' && typeof expected === 'number')
    return actual === expected ? 1 : Math.max(0, 1 - Math.abs(actual - expected) / Math.max(Math.abs(actual), Math.abs(expected)));
  return canonical(actual) === canonical(expected) ? 1 : 0;
}

/**
 * How a folder's result files are checked (semantics.files_oracle). The items are the files the task should change and
 * the files that did change (a file changed or made that the task does not is a failed item), or with `csv` the rows of
 * the changed files; at least `threshold` (default 0.9) of them must pass.
 * - exact: the same content.
 * - rewrite: the same front matter, a body changed from the input's and still about it (token F1 at least 0.3); for
 *   rewrites that many wordings satisfy, where a reference rewrite is one of them.
 * - csv: rows keyed by their first cell; each expected row's cells match (both empty, or token F1 at least `span`,
 *   default 0.5), and the header is the same.
 * - counts: "label: count" lines, scored as a record of counts (agreement).
 */
export type FilesOracle = { compare?: 'exact' | 'rewrite' | 'csv' | 'counts'; threshold?: number; span?: number };
export type FilesVerdict = { accepted: boolean; score: number; passed: number; items: number; failed: string[] };

const FRONT_MATTER = /^---\n[\s\S]*?\n---\n/;
function csvRows(text: string): string[][] {
  const rows: string[][] = []; let row: string[] = [], cell = '', quoted = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!;
    if (quoted) { if (char === '"' && text[index + 1] === '"') { cell += '"'; index++; } else if (char === '"') quoted = false; else cell += char; }
    else if (char === '"') quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else if (char !== '\r') cell += char;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
const countLines = (text: string) => Object.fromEntries(text.split('\n').map(line => /^(.+?):\s*(-?\d+)\s*$/.exec(line.trim()))
  .filter(match => match).map(match => [match![1]!.trim(), Number(match![2])]));

export function checkFiles(actual: Record<string, string>, expected: Record<string, string>, input: Record<string, string>,
    spec: FilesOracle = {}): FilesVerdict {
  const compare = spec.compare ?? 'exact', threshold = spec.threshold ?? 0.9, span = spec.span ?? 0.5;
  const paths = [...new Set([...Object.keys(expected), ...Object.keys(actual), ...Object.keys(input)])].sort()
    .filter(path => expected[path] !== input[path] || actual[path] !== input[path]);
  let passed = 0, items = 0;
  const failed: string[] = [];
  const item = (name: string, ok: boolean) => { items++; if (ok) passed++; else failed.push(name); };
  for (const path of paths) {
    const got = actual[path], want = expected[path];
    if (got === undefined || want === undefined) { item(path, got === want); continue; }
    if (compare === 'csv') {
      const [head, ...rows] = csvRows(want), [gotHead, ...gotRows] = csvRows(got);
      const byKey = new Map(gotRows.map(row => [row[0], row]));
      const same = canonical(head) === canonical(gotHead);
      for (const row of rows) {
        const other = byKey.get(row[0]); byKey.delete(row[0]);
        item(`${path}:${row[0]}`, same && !!other && row.every((cell, index) => {
          const theirs = other[index] ?? '';
          return !cell.trim() || !theirs.trim() ? !cell.trim() && !theirs.trim() : spanF1(theirs, cell) >= span;
        }));
      }
      for (const key of byKey.keys()) item(`${path}:${key}`, false);
    } else if (compare === 'counts') item(path, agreement(countLines(got), countLines(want)) >= threshold);
    else if (compare === 'rewrite' && input[path] !== undefined) {
      const front = (text: string) => FRONT_MATTER.exec(text)?.[0] ?? '', body = (text: string) => text.slice(front(text).length);
      item(path, front(got) === front(want) && body(got).trim() !== body(input[path]!).trim() &&
        spanF1(body(got), body(input[path]!)) >= 0.3);
    } else item(path, got === want);
  }
  const score = items ? passed / items : 1;
  return { accepted: score >= threshold, score, passed, items, failed: failed.slice(0, 20) };
}

/** A judge's verdict is supplied by the collection caller; no case can self-certify its own prose. */
export async function checkOracle(actual: unknown, expected: unknown, oracle: OracleSpec = 'exact',
  judge?: (input: { actual: unknown; expected: unknown; rubric: string }) => Promise<{ accepted: boolean; verdict: string }>):
  Promise<OracleVerdict> {
  const spec = typeof oracle === 'string' ? { level: oracle } : oracle;
  const { level } = spec;
  if (!ORACLE_LEVELS.includes(level)) throw new RangeError(`unknown oracle level: ${level}`);
  const candidates = [expected, ...('alternates' in spec ? spec.alternates ?? [] : [])];
  if (level === 'exact') return { accepted: candidates.some(candidate => canonical(actual) === canonical(candidate)), level };
  if (level === 'normalized') return { accepted: candidates.some(candidate => normalized(actual) === normalized(candidate)), level };
  if (level === 'agreement') {
    const threshold = 'threshold' in spec ? spec.threshold ?? 0.9 : 0.9;
    const score = Math.max(...candidates.map(candidate => agreement(actual, candidate)));
    return { accepted: score >= threshold, level, score };
  }
  if (level === 'span') {
    const threshold = 'threshold' in spec ? spec.threshold ?? 0.5 : 0.5;
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('span threshold must be between 0 and 1');
    const score = Math.max(...candidates.map(candidate => spanF1(actual, candidate)));
    return { accepted: score >= threshold, level, score };
  }
  if (!judge) return { accepted: false, level, verdict: 'no judge supplied' };
  const verdict = await judge({ actual, expected, rubric: 'rubric' in spec ? spec.rubric ?? '' : '' });
  return { accepted: verdict.accepted === true, level, verdict: verdict.verdict };
}
