import { checkConstraints, type WritingConstraint } from './constraints.js';
/** Answer checks used by generated and dataset-backed teacher cases. */
export type OracleLevel = 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' | 'constraints';
export const ORACLE_LEVELS: readonly OracleLevel[] = ['exact', 'normalized', 'span', 'agreement', 'judged', 'constraints'];
/** Bump whenever answer comparison semantics change so older outcomes cannot stand in for new runs. */
export const ANSWER_COMPARISON_VERSION = 'normalized-decimal-exact/2';
export type OracleSpec = OracleLevel | { level: OracleLevel; alternates?: unknown[];
  threshold?: number; normalization?: 'qa' | 'qa-string-map' | 'named-tree' | 'json-string-record' | 'tatqa-answer-record' | 'tatqa-answer-record-exact'; rubric?: string; context?: unknown; [key: string]: unknown };
export type OracleVerdict = { accepted: boolean; level: OracleLevel; score?: number; verdict?: string; needs_review?: boolean };

/** TreeDST's author implementation keys children by name; sibling order is not semantic.
 * Reject duplicate names, extra fields and malformed nodes instead of silently dropping them.
 */
export function namedTreeCanonical(value: unknown): string | null {
  let remaining = 10000;
  const visit = (input: unknown, depth: number): unknown => {
    if (--remaining < 0 || depth > 128 || !input || typeof input !== 'object' || Array.isArray(input)) throw new Error('invalid_named_tree');
    const node = input as Record<string, unknown>;
    if (typeof node.name !== 'string' || !Array.isArray(node.children) ||
        Object.keys(node).length !== 2 || !Object.hasOwn(node, 'name') || !Object.hasOwn(node, 'children')) throw new Error('invalid_named_tree');
    const names = new Set<string>();
    const children = node.children.map(child => {
      const parsed = visit(child, depth + 1) as [string, unknown[]];
      if (names.has(parsed[0])) throw new Error('duplicate_named_child');
      names.add(parsed[0]); return parsed;
    }).sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
    return [node.name, children];
  };
  try { return JSON.stringify(visit(value, 0)); } catch { return null; }
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}`;
  return JSON.stringify(value);
}
/** Formatting is irrelevant for JSON string records; duplicate keys and other shapes are invalid. */
export function jsonStringRecordCanonical(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const string = String.raw`"(?:[^"\\\u0000-\u001f]|\\(?:["\\/bfnrt]|u[\da-fA-F]{4}))*"`;
  const pair = `${string}\\s*:\\s*${string}`;
  if (!new RegExp(`^\\s*\\{\\s*(?:${pair}(?:\\s*,\\s*${pair})*)?\\s*\\}\\s*$`).test(value)) return null;
  try {
    const names = new Set<string>();
    for (const match of value.matchAll(new RegExp(`(${string})\\s*:\\s*${string}`, 'g'))) {
      const key = JSON.parse(match[1]!) as string;
      if (names.has(key)) return null;
      names.add(key);
    }
    return canonical(JSON.parse(value));
  } catch { return null; }
}

const TATQA_SCALES = new Set(['', 'percent', 'thousand', 'million', 'billion']);
const TATQA_NUMERIC = /^[+-]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)$/;

/** Canonical exact decimal, plus an optional 2dp form when rounding is unambiguous. */
function tatqaNumberForms(value: string): { exact: string; rounded: string | null } | null {
  if (value.length > 512 || !TATQA_NUMERIC.test(value)) return null;
  const negative = value.startsWith('-');
  const unsigned = value.replace(/^[+-]/, '').replace(/,/g, '');
  const [wholeRaw = '0', fractionRaw = ''] = unsigned.split('.');
  const whole = wholeRaw || '0';
  const digits = `${whole}${fractionRaw}`.replace(/^0+(?=\d)/, '');
  // Equivalent decimal encodings such as 12, 12.0 and +12.00 share one exact form.
  const normalizedExact = (() => {
    let d = digits, scale = fractionRaw.length;
    while (scale > 0 && d.endsWith('0')) { d = d.slice(0, -1); scale--; }
    if (!d) d = '0';
    return `${negative && d !== '0' ? '-' : ''}${d}:${scale}`;
  })();
  const magnitude = BigInt(digits || '0');
  if (fractionRaw.length <= 2) {
    const cents = magnitude * 10n ** BigInt(2 - fractionRaw.length);
    return { exact: normalizedExact, rounded: `${negative && cents !== 0n ? '-' : ''}${cents}` };
  }
  const scaleFactor = 10n ** BigInt(fractionRaw.length - 2);
  let cents = magnitude / scaleFactor;
  const remainder = magnitude % scaleFactor;
  const half = scaleFactor / 2n;
  // Exclude exact half-cent ties from rounded equality; exact decimal equality remains valid.
  const tie = remainder === half;
  if (remainder > half) cents++;
  return { exact: normalizedExact,
    rounded: tie ? null : `${negative && cents !== 0n ? '-' : ''}${cents}` };
}

/** TaTQA-only comparison: strict record schema and scale, with source scorer's numeric rounding. */
export function tatqaAnswerRecordCanonical(value: unknown): string | null {
  const recordText = jsonStringRecordCanonical(value);
  if (recordText === null || typeof value !== 'string') return null;
  try {
    const record = JSON.parse(value) as Record<string, unknown>;
    if (Object.keys(record).length !== 2 || !Object.hasOwn(record, 'answer') || !Object.hasOwn(record, 'scale') ||
        typeof record.answer !== 'string' || typeof record.scale !== 'string' || !TATQA_SCALES.has(record.scale)) return null;
    const numeric = tatqaNumberForms(record.answer);
    return JSON.stringify({ answer: numeric ? { numeric: true, exact: numeric.exact, rounded: numeric.rounded } :
      { numeric: false, text: record.answer },
      scale: record.scale });
  } catch { return null; }
}

export function tatqaAnswerRecordsEqual(actual: unknown, expected: unknown, numericComparison: 'rounded-2dp' | 'exact' = 'rounded-2dp'): boolean {
  if (typeof actual !== 'string' || typeof expected !== 'string') return false;
  const left = tatqaAnswerRecordCanonical(actual), right = tatqaAnswerRecordCanonical(expected);
  if (left === null || right === null) return false;
  if (left === right) return true;
  try {
    const a = JSON.parse(left) as { answer: { numeric: boolean; exact?: string; rounded?: string | null; text?: string }; scale: string };
    const b = JSON.parse(right) as { answer: { numeric: boolean; exact?: string; rounded?: string | null; text?: string }; scale: string };
    if (a.scale !== b.scale || !a.answer.numeric || !b.answer.numeric) return false;
    if (numericComparison === 'exact') return a.answer.exact === b.answer.exact;
    return a.answer.exact === b.answer.exact || (!!a.answer.rounded && !!b.answer.rounded &&
      a.answer.rounded === b.answer.rounded);
  } catch { return false; }
}
function normalizeText(value: unknown): string {
  const text = String(value ?? '').normalize('NFKC').trim().toLocaleLowerCase('en-US').replace(/\s+/g, ' ');
  const number = text.replace(/(?<=\d),(?=\d{3}(?:\D|$))/g, '');
  if (number.length <= 512 && /^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(number)) {
    // Reuse only TaTQA's exact decimal representation; generic normalized answers must never
    // inherit its optional source-specific two-decimal rounding equivalence.
    const exact = tatqaNumberForms(number)?.exact ?? (number.endsWith('.') ? tatqaNumberForms(`${number}0`)?.exact : undefined);
    if (exact !== undefined) return `number:${exact}`;
  }
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
function qaAnswerText(value: string): string {
  const normalized = value.normalize('NFKC').toLocaleLowerCase('en-US')
    .replace(/[\p{P}\p{S}]/gu, '');
  return normalized.split(/\s+/u).filter(token => token && !['a', 'an', 'the'].includes(token)).join(' ');
}
/** Canonical map for extractive QA: keys stay exact; answer values use SQuAD token normalization. */
export function qaStringMapCanonical(value: unknown): string | null {
  let record: unknown = value;
  if (typeof value === 'string') {
    if (jsonStringRecordCanonical(value) === null) return null;
    try { record = JSON.parse(value); } catch { return null; }
  }
  if (!record || typeof record !== 'object' || Array.isArray(record)) return null;
  const entries = Object.entries(record as Record<string, unknown>);
  if (!entries.every(([, answer]) => typeof answer === 'string')) return null;
  return JSON.stringify(entries.sort(([a], [b]) => a.localeCompare(b))
    .map(([key, answer]) => [key, qaAnswerText(answer as string)]));
}
function words(value: unknown, qa = false): string[] {
  const tokens = String(Array.isArray(value) ? value.join(' ') : value ?? '').normalize('NFKC')
    .toLocaleLowerCase('en-US').match(/[\p{L}\p{N}]+/gu) ?? [];
  return qa ? tokens.filter(word => !['a', 'an', 'the'].includes(word)) : tokens;
}
/** Multiset token F1; repeated words count only when they occur on both sides. */
export function spanF1(actual: unknown, expected: unknown, qa = false): number {
  if (qa) {
    const numbers = (value: unknown) => String(value).match(/[+-]?\d+(?:\.\d+)?/g)?.map(token =>
      tatqaNumberForms(token)?.exact ?? token).sort() ?? [];
    const goldNumbers = numbers(expected), gotNumbers = numbers(actual);
    if (goldNumbers.length && canonical(goldNumbers) !== canonical(gotNumbers)) return 0;
  }
  const left = words(actual, qa), right = words(expected, qa);
  if (qa && [actual, expected].some(value => ['yes', 'no', 'noanswer'].includes(String(value).trim().toLowerCase())) &&
      canonical(left) !== canonical(right)) return 0;
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
  const invalid = (value: unknown): boolean => typeof value === 'number' ? !Number.isFinite(value) :
    !!value && typeof value === 'object' && Object.values(value).some(invalid);
  if (invalid(actual) || invalid(expected)) return 0;
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

/** File contracts fail closed on malformed reports and unverified rewrites. */
export const DATA_QUALITY_VERSION = 2;
export const FILE_CONTENT_COMPARISON_VERSION = 'json-content/2';
export type FilesOracle = { compare?: 'content' | 'exact' | 'moves' | 'rewrite' | 'csv' | 'counts' | 'json-string-record' | 'qa-string-map' | 'tatqa-answer-record' | 'tatqa-answer-record-exact' | 'markdown-terminal-newline'; threshold?: number; span?: number;
  total?: number; rubric?: string; alternates?: Record<string, string[]>;
  /** Explicitly allowlisted Markdown files whose one terminal line ending may vary. */
  markdown_terminal_newline_paths?: string[];
  /** Reports may only quote the corresponding original source. */
  quote_sources?: Record<string, string>;
  /** Alternate annotated clauses, keyed by the report row's id. */
  quote_alternates?: Record<string, string[]>;
  return_count?: 'changed' | 'csv_nonempty' | 'counts'; report?: string };
export type FilesVerdict = { accepted: boolean; score: number; passed: number; items: number; failed: string[];
  comparison_version?: string;
  quality_version: number; errors: string[]; pending: string[]; positive_recall?: number; positive_precision?: number;
  judgments?: Record<string, { accepted: boolean; verdict: string; needs_review?: boolean }> };
export type OracleJudge = (input: { actual: unknown; expected: unknown; rubric: string }) =>
  Promise<{ accepted: boolean; verdict: string; needs_review?: boolean }>;
const FRONT_MATTER = /^---\r?\n[\s\S]*?\r?\n---\r?\n/;
const front = (text: string) => FRONT_MATTER.exec(text)?.[0] ?? '';
const body = (text: string) => text.slice(front(text).length);
const probability = (value: number, name: string) => {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new RangeError(`${name} threshold must be between 0 and 1`);
  return value;
};

/** Strict CSV, including multiline quoted cells. Never silently overwrite duplicate row ids. */
export function csvRows(text: string): string[][] {
  const rows: string[][] = []; let row: string[] = [], cell = '', quoted = false, closed = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!;
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { cell += '"'; index++; }
      else if (char === '"') { quoted = false; closed = true; } else cell += char;
    } else if (char === ',') { row.push(cell); cell = ''; closed = false; }
    else if (char === '\n' || char === '\r' && text[index + 1] === '\n') {
      if (char === '\r') index++;
      row.push(cell); rows.push(row); row = []; cell = ''; closed = false;
    } else if (char === '"' && !cell && !closed) quoted = true;
    else if (char === '"' || closed || char === '\r') throw new Error('malformed CSV quoting');
    else cell += char;
  }
  if (quoted) throw new Error('unterminated CSV quote');
  if (cell || row.length || closed) { row.push(cell); rows.push(row); }
  if (!rows.length || !rows[0]!.length) throw new Error('missing CSV header');
  const keys = new Set<string>();
  for (const row of rows.slice(1)) {
    if (row.length !== rows[0]!.length) throw new Error('CSV row width differs from header');
    if (!row[0]?.trim() || keys.has(row[0])) throw new Error('empty or duplicate CSV row id');
    keys.add(row[0]);
  }
  return rows;
}
export function countLines(text: string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const line of text.split(/\r?\n/).filter(line => line.trim())) {
    const match = /^(.+?):\s*(\d+)\s*$/.exec(line.trim());
    if (!match || Object.hasOwn(counts, match[1]!.trim()) || !Number.isSafeInteger(Number(match[2])))
      throw new Error('invalid or duplicate count line');
    counts[match[1]!.trim()] = Number(match[2]);
  }
  return counts;
}
const quoteText = (text: string) => text.normalize('NFKC').replace(/\s+/g, ' ').trim();

/** JSON formatting is not data. Preserve strings, types, array order and every key;
 * reject duplicate keys and unsafe numeric values rather than silently losing data.
 */
export function jsonFileCanonical(text: string): string | null {
  try {
    const value: unknown = JSON.parse(text);
    const stack: (Set<string> | null)[] = [];
    const tokens = [...text.matchAll(/"(?:\\.|[^"\\])*"|[{}\[\]:,]/g)].map(match => match[0]);
    for (let index = 0; index < tokens.length; index++) {
      const token = tokens[index]!;
      if (token === '{') stack.push(new Set());
      else if (token === '[') stack.push(null);
      else if (token === '}' || token === ']') stack.pop();
      else if (token.startsWith('"') && tokens[index + 1] === ':') {
        const keys = stack.at(-1), key = JSON.parse(token) as string;
        if (!keys || keys.has(key)) return null;
        keys.add(key);
      }
    }
    const valid = (part: unknown): boolean => typeof part === 'number' ?
      Number.isFinite(part) && (!Number.isInteger(part) || Number.isSafeInteger(part)) :
      part !== null && typeof part === 'object' ? Object.values(part).every(valid) : true;
    return valid(value) ? canonical(value) : null;
  } catch { return null; }
}

export function fileContentEqual(path: string, actual: string | undefined, expected: string | undefined): boolean {
  if (actual === undefined || expected === undefined) return actual === expected;
  if (!/\.json$/i.test(path)) return actual === expected;
  const parsed = jsonFileCanonical(actual);
  return parsed !== null && parsed === jsonFileCanonical(expected);
}

/** Remove at most one terminal LF or CRLF. No other whitespace or line-ending normalization occurs. */
export function markdownTerminalNewlineBody(text: string): string {
  return text.endsWith('\r\n') ? text.slice(0, -2) : text.endsWith('\n') ? text.slice(0, -1) : text;
}
export function markdownTerminalNewlineEqual(path: string, actual: string | undefined, expected: string | undefined): boolean {
  return typeof actual === 'string' && typeof expected === 'string' && /\.md$/i.test(path) &&
    markdownTerminalNewlineBody(actual) === markdownTerminalNewlineBody(expected);
}

export function checkFiles(actual: Record<string, string>, expected: Record<string, string>, input: Record<string, string>,
    spec: FilesOracle = {}, judgments: FilesVerdict['judgments'] = {}): FilesVerdict {
  const compare = spec.compare ?? 'exact', threshold = probability(spec.threshold ?? 0.9, 'files'),
    span = probability(spec.span ?? 0.5, 'files span');
  if (compare === 'content') {
    const paths = [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort();
    const failed = paths.filter(path => !fileContentEqual(path, actual[path], expected[path]));
    // Structural file contracts always require every file; no partial-credit threshold.
    return { accepted: !failed.length, score: paths.length ? (paths.length - failed.length) / paths.length : 1,
      passed: paths.length - failed.length, items: paths.length, failed, errors: [], pending: [],
      quality_version: DATA_QUALITY_VERSION };
  }
  if (compare === 'markdown-terminal-newline') {
    const allow = spec.markdown_terminal_newline_paths;
    const validAllow = Array.isArray(allow) && allow.length > 0 && new Set(allow).size === allow.length &&
      allow.every(path => typeof path === 'string' && path.length > 0 && !path.startsWith('/') &&
        !path.includes('\\') && !path.split('/').some(part => !part || part === '.' || part === '..') && /\.md$/i.test(path));
    if (!validAllow) return { accepted:false, score:0, passed:0, items:0, failed:[], errors:['invalid_markdown_terminal_newline_allowlist'], pending:[], quality_version:DATA_QUALITY_VERSION };
    const allowed = new Set(allow);
    const paths = [...new Set([...Object.keys(expected), ...Object.keys(actual), ...Object.keys(input)])].sort();
    const failed: string[] = [], errors: string[] = [];
    for (const path of allowed) {
      if (typeof input[path] !== 'string' || typeof expected[path] !== 'string')
        errors.push(`missing_markdown_target:${path}`);
    }
    let passed = 0;
    for (const path of paths) {
      const got = actual[path], want = expected[path], before = input[path];
      const ok = allowed.has(path) ? markdownTerminalNewlineEqual(path, got, want) : got === want && want === before;
      if (allowed.has(path)) {
        if (before === undefined || want === undefined || got === undefined ||
            markdownTerminalNewlineEqual(path, before, want)) errors.push(`invalid_or_vacuous_markdown_target:${path}`);
      } else if (want !== before) errors.push(`unexpected_expected_change:${path}`);
      if (want === undefined || got === undefined || before === undefined) {
        if (!(want === undefined && got === undefined && before === undefined)) errors.push(`missing_or_unexpected_file:${path}`);
      }
      if (ok) passed++; else failed.push(path);
    }
    const score = paths.length ? passed / paths.length : 1;
    return { accepted: !errors.length && !failed.length && score >= threshold, score, passed, items:paths.length,
      failed, errors, pending:[], quality_version:DATA_QUALITY_VERSION };
  }
  const paths = [...new Set([...Object.keys(expected), ...Object.keys(actual), ...Object.keys(input)])].sort()
    .filter(path => expected[path] !== input[path] || actual[path] !== input[path]);
  let passed = 0, items = 0, goldPositive = 0, actualPositive = 0, matchedPositive = 0;
  const failed: string[] = [], errors: string[] = [], pending: string[] = [];
  const item = (name: string, ok: boolean) => { items++; if (ok) passed++; else failed.push(name); };
  if (compare === 'moves') {
    const covered = new Set<string>();
    for (const [source, original] of Object.entries(input)) {
      const name = source.split('/').pop();
      const destinations = Object.keys(expected).filter(path => path.split('/').pop() === name);
      const gotPaths = Object.keys(actual).filter(path => path.split('/').pop() === name);
      gotPaths.forEach(path => covered.add(path));
      const destination = destinations[0];
      item(source, destinations.length === 1 && gotPaths.length === 1 && gotPaths[0] === destination && actual[destination!] === original);
      if (gotPaths.length !== 1 || gotPaths.some(path => actual[path] !== original)) errors.push(`lost_or_corrupted_file:${source}`);
    }
    for (const path of Object.keys(actual)) if (!covered.has(path)) errors.push(`unexpected_file:${path}`);
    const score = items ? passed / items : 1;
    return { accepted: !errors.length && score >= threshold, score, passed, items, failed, errors, pending, quality_version: DATA_QUALITY_VERSION };
  }
  for (const path of paths) {
    const got = actual[path], want = expected[path];
    // Read-only source corruption and unexpected output are contract failures, never diluted by many report rows.
    if (want === input[path] && got !== want || want === undefined && got !== undefined) errors.push(`unexpected_change:${path}`);
    if (got === undefined || want === undefined) { item(path, got === want); continue; }
    // Existing source files are checked exactly, even in a CSV/rewrite task.
    if (want === input[path]) { item(path, got === want); continue; }
    if (compare === 'json-string-record') {
      const parsed = jsonStringRecordCanonical(got);
      const candidates = [want, ...(spec.alternates?.[path] ?? [])];
      item(path, parsed !== null && candidates.some(candidate => parsed === jsonStringRecordCanonical(candidate)));
      continue;
    }
    if (compare === 'qa-string-map') {
      const parsed = qaStringMapCanonical(got);
      item(path, path === 'answers.json' && parsed !== null && parsed === qaStringMapCanonical(want));
      continue;
    }
    if (compare === 'tatqa-answer-record' || compare === 'tatqa-answer-record-exact') {
      item(path, path === 'answer.json' && tatqaAnswerRecordsEqual(got, want,
        compare === 'tatqa-answer-record-exact' ? 'exact' : 'rounded-2dp'));
      continue;
    }
    if (compare === 'csv') {
      try {
        const [head, ...rows] = csvRows(want), [gotHead, ...gotRows] = csvRows(got);
        if (canonical(head) !== canonical(gotHead)) throw new Error('CSV header mismatch');
        const byKey = new Map(gotRows.map(row => [row[0], row]));
        if (!rows.length) item(path, !gotRows.length);
        for (const row of gotRows) if (row[1]?.trim()) actualPositive++;
        for (const row of rows) {
          const key = row[0]!, other = byKey.get(key); byKey.delete(key);
          const positive = !!row[1]?.trim(); if (positive) goldPositive++;
          const ok = !!other && row.every((cell, index) => {
            const theirs = other[index] ?? '';
            if (!cell.trim() || !theirs.trim()) return !cell.trim() && !theirs.trim();
            if (index === 0) return theirs === cell;
            const candidates = index === 1 ? [cell, ...(spec.quote_alternates?.[key] ?? [])] : [cell];
            if (index === 1 && spec.quote_sources) return candidates.some(candidate => quoteText(theirs) === quoteText(candidate)) ||
              judgments[`${path}:${key}`]?.accepted === true && !judgments[`${path}:${key}`]?.needs_review;
            return candidates.some(candidate => spanF1(theirs, candidate) >= span);
          });
          if (positive && ok) matchedPositive++;
          item(`${path}:${key}`, ok);
        }
        for (const key of byKey.keys()) item(`${path}:${key}`, false);
        if (spec.quote_sources) for (const row of gotRows) {
          if (!row[1]?.trim()) continue;
          const source = input[spec.quote_sources[row[0]!] ?? ''];
          if (!source || !quoteText(body(source)).includes(quoteText(row[1]))) errors.push(`quote_not_in_source:${row[0]}`);
          const gold = rows.find(candidate => candidate[0] === row[0])?.[1] ?? '';
          if (![gold, ...(spec.quote_alternates?.[row[0]!] ?? [])].some(candidate => quoteText(candidate) === quoteText(row[1]!)) &&
              (judgments[`${path}:${row[0]}`]?.accepted !== true || judgments[`${path}:${row[0]}`]?.needs_review)) {
            if (!judgments[`${path}:${row[0]}`] || judgments[`${path}:${row[0]}`]?.needs_review) pending.push(`unverified_quote:${row[0]}`);
            else errors.push(`invalid_quote:${row[0]}`);
          }
        }
      } catch (error) { errors.push(`${path}:${(error as Error).message}`); item(path, false); }
    } else if (compare === 'counts') {
      try {
        const counts = countLines(got);
        if (spec.total !== undefined && Object.values(counts).reduce((sum, count) => sum + count, 0) !== spec.total)
          errors.push(`wrong_total:${path}`);
        item(path, agreement(counts, countLines(want)) >= threshold);
      }
      catch (error) { errors.push(`${path}:${(error as Error).message}`); item(path, false); }
    } else if (compare === 'rewrite' && input[path] !== undefined) {
      const validShape = front(got) === front(input[path]!) && front(got) === front(want);
      const referenceMatch = [want, ...(spec.alternates?.[path] ?? [])].some(candidate => body(got).trim() === body(candidate).trim());
      if (validShape && !referenceMatch && (!judgments[path] || judgments[path]?.needs_review)) pending.push(`unverified_rewrite:${path}`);
      item(path, validShape && (referenceMatch || judgments[path]?.accepted === true && !judgments[path]?.needs_review));
    } else item(path, got === want);
  }
  const score = items ? passed / items : 1;
  const positiveRecall = goldPositive ? matchedPositive / goldPositive : 1;
  const positivePrecision = actualPositive ? matchedPositive / actualPositive : goldPositive ? 0 : 1;
  return { accepted: !errors.length && !pending.length && score >= threshold && (compare !== 'rewrite' || score === 1) && (compare !== 'csv' ||
      positiveRecall >= threshold && positivePrecision >= threshold), score, passed, items, failed: failed.slice(0, 20),
    errors, pending, quality_version: DATA_QUALITY_VERSION,
    ...(compare === 'qa-string-map' ? { comparison_version: 'squad-token-map/1' } : {}),
    ...(compare === 'csv' ? { positive_recall: positiveRecall, positive_precision: positivePrecision } : {}),
    ...(Object.keys(judgments).length ? { judgments } : {}) };
}

/** Non-reference rewrites require a separately configured judge; a word-overlap heuristic cannot certify them. */
export async function checkFilesWithJudge(actual: Record<string, string>, expected: Record<string, string>,
    input: Record<string, string>, spec: FilesOracle = {}, judge?: OracleJudge): Promise<FilesVerdict> {
  const first = checkFiles(actual, expected, input, spec);
  if (!['rewrite', 'csv'].includes(spec.compare ?? '') || !judge || !spec.rubric) return first;
  const judgments: NonNullable<FilesVerdict['judgments']> = {};
  for (const path of Object.keys(expected)) {
    if (spec.compare === 'csv') {
      if (!spec.quote_sources || expected[path] === input[path] || actual[path] === undefined) continue;
      let rows: string[][], goldRows: string[][];
      try { rows = csvRows(actual[path]!); goldRows = csvRows(expected[path]!); } catch { continue; }
      for (const row of rows.slice(1)) {
        if (!row[1]?.trim()) continue;
        const gold = goldRows.slice(1).find(candidate => candidate[0] === row[0])?.[1] ?? '';
        if ([gold, ...(spec.quote_alternates?.[row[0]!] ?? [])].some(candidate => quoteText(candidate) === quoteText(row[1]!))) continue;
        const source = input[spec.quote_sources[row[0]!] ?? ''];
        if (!source || !quoteText(body(source)).includes(quoteText(row[1]))) continue;
        const sourceBody = body(source), location = quoteText(sourceBody).indexOf(quoteText(row[1]));
        // Keep grading within a small context. The grader must defer if omitted material is needed.
        const rawLocation = sourceBody.indexOf(row[1]);
        const sourceView = sourceBody.length <= 12000 ? { full_text: sourceBody } :
          { opening: sourceBody.slice(0, 1500), quote_context: rawLocation >= 0 ?
            sourceBody.slice(Math.max(0, rawLocation - 2000), rawLocation + row[1].length + 2000) :
            quoteText(sourceBody).slice(Math.max(0, location - 2000), location + quoteText(row[1]).length + 2000),
            omitted_characters: Math.max(0, sourceBody.length - 1500 - row[1].length - 4000) };
        judgments[`${path}:${row[0]}`] = await judge({ actual: { source_view: sourceView, clause: row[1] },
          expected: { annotated_clauses: [gold, ...(spec.quote_alternates?.[row[0]!] ?? [])] }, rubric: spec.rubric });
      }
      continue;
    }
    if (actual[path] === undefined || input[path] === undefined || expected[path] === input[path] ||
        front(actual[path]!) !== front(input[path]!)) continue;
    if ([expected[path]!, ...(spec.alternates?.[path] ?? [])].some(value => body(actual[path]!).trim() === body(value).trim())) continue;
    judgments[path] = await judge({ actual: { original: body(input[path]!), revised: body(actual[path]!) },
      expected: { reference: body(expected[path]!) }, rubric: spec.rubric });
  }
  return checkFiles(actual, expected, input, spec, judgments);
}

/** Returned summaries must describe the actual output, independently of tolerance against noisy gold. */
export function fileReturnValue(files: Record<string, string>, input: Record<string, string>, spec: FilesOracle): unknown {
  if (spec.return_count === 'changed') return Object.keys(files).filter(path => spec.compare === 'content' ?
    !fileContentEqual(path, files[path], input[path]) : files[path] !== input[path]).length;
  const report = files[spec.report ?? '']; if (report === undefined) throw new Error('missing report');
  if (spec.return_count === 'counts') return countLines(report);
  return csvRows(report).slice(1).filter(row => row[1]?.trim()).length;
}
export function checkFileReturn(actual: unknown, files: Record<string, string>, input: Record<string, string>, spec: FilesOracle): boolean {
  if (spec.compare === 'tatqa-answer-record') {
    const written = files['answer.json'];
    return written !== undefined && tatqaAnswerRecordsEqual(actual, written);
  }
  if (spec.compare === 'tatqa-answer-record-exact') {
    const written = files['answer.json'];
    return written !== undefined && tatqaAnswerRecordsEqual(actual, written, 'exact');
  }
  if (spec.compare === 'json-string-record') {
    const written = files['answer.json'];
    const returned = jsonStringRecordCanonical(actual);
    return written !== undefined && returned !== null && returned === jsonStringRecordCanonical(written);
  }
  if (!spec.return_count) return true;
  try { return canonical(actual) === canonical(fileReturnValue(files, input, spec)); } catch { return false; }
}

/** A judge's verdict is supplied by the collection caller; no case can self-certify its own prose. */
export async function checkOracle(actual: unknown, expected: unknown, oracle: OracleSpec = 'exact',
  judge?: (input: { actual: unknown; expected: unknown; rubric: string }) => Promise<{ accepted: boolean; verdict: string; needs_review?: boolean }>):
  Promise<OracleVerdict> {
  const spec = typeof oracle === 'string' ? { level: oracle } : oracle;
  const { level } = spec;
  if (!ORACLE_LEVELS.includes(level)) throw new RangeError(`unknown oracle level: ${level}`);
  const candidates = [expected, ...('alternates' in spec ? spec.alternates ?? [] : [])];
  if (level === 'exact') return { accepted: candidates.some(candidate => canonical(actual) === canonical(candidate)), level };
  if (level === 'constraints') {
    // `expected` is the list of verifiable writing constraints; every one must hold (evaluation/constraints.ts).
    const result = checkConstraints(actual, expected as WritingConstraint[]);
    return { accepted: result.passed, level, score: result.score, ...(result.failed.length ? { verdict: result.failed.join(' ') } : {}) };
  }
  if (level === 'normalized' && spec.normalization === 'named-tree') {
    const answer = namedTreeCanonical(actual);
    return { accepted: answer !== null && candidates.some(candidate => namedTreeCanonical(candidate) === answer), level };
  }
  if (level === 'normalized' && spec.normalization === 'qa-string-map') {
    const answer = qaStringMapCanonical(actual);
    return { accepted: answer !== null && candidates.some(candidate => qaStringMapCanonical(candidate) === answer), level };
  }
  if (level === 'normalized' && spec.normalization === 'json-string-record') {
    const answer = jsonStringRecordCanonical(actual);
    return { accepted: answer !== null && candidates.some(candidate => jsonStringRecordCanonical(candidate) === answer), level };
  }
  if (level === 'normalized' && (spec.normalization === 'tatqa-answer-record' || spec.normalization === 'tatqa-answer-record-exact')) {
    return { accepted: candidates.some(candidate => tatqaAnswerRecordsEqual(actual, candidate,
      spec.normalization === 'tatqa-answer-record-exact' ? 'exact' : 'rounded-2dp')), level };
  }
  if (level === 'normalized') return { accepted: candidates.some(candidate => normalized(actual) === normalized(candidate)), level };
  if (level === 'agreement') {
    const threshold = probability('threshold' in spec ? spec.threshold ?? 0.9 : 0.9, 'agreement');
    const score = Math.max(...candidates.map(candidate => agreement(actual, candidate)));
    return { accepted: score >= threshold, level, score };
  }
  if (level === 'span') {
    const threshold = 'threshold' in spec ? spec.threshold ?? 0.5 : 0.5;
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('span threshold must be between 0 and 1');
    const score = Math.max(...candidates.map(candidate => spanF1(actual, candidate, 'normalization' in spec && spec.normalization === 'qa')));
    if (score < threshold && !judge && 'rubric' in spec && spec.rubric)
      return { accepted: false, level, score, needs_review: true, verdict: 'answer equivalence needs independent review' };
    if (score < threshold && judge && 'rubric' in spec && spec.rubric) {
      const verdict = await judge({ actual, expected: { answer: expected, context: spec.context }, rubric: spec.rubric });
      return { accepted: verdict.accepted === true && !verdict.needs_review, level, score, verdict: verdict.verdict, ...(verdict.needs_review ? { needs_review: true } : {}) };
    }
    return { accepted: score >= threshold, level, score };
  }
  if (!judge) return { accepted: false, level, verdict: 'no judge supplied', needs_review: true };
  const verdict = await judge({ actual, expected, rubric: 'rubric' in spec ? spec.rubric ?? '' : '' });
  return { accepted: verdict.accepted === true && !verdict.needs_review, level, verdict: verdict.verdict, ...(verdict.needs_review ? { needs_review: true } : {}) };
}
