/** Answer checks used by generated and dataset-backed teacher cases. */
export type OracleLevel = 'exact' | 'normalized' | 'span' | 'agreement' | 'judged';
export const ORACLE_LEVELS: readonly OracleLevel[] = ['exact', 'normalized', 'span', 'agreement', 'judged'];
export type OracleSpec = OracleLevel | { level: OracleLevel; alternates?: unknown[];
  threshold?: number; normalization?: 'qa'; rubric?: string; context?: unknown; [key: string]: unknown };
export type OracleVerdict = { accepted: boolean; level: OracleLevel; score?: number; verdict?: string; needs_review?: boolean };

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
function words(value: unknown, qa = false): string[] {
  const tokens = String(Array.isArray(value) ? value.join(' ') : value ?? '').normalize('NFKC')
    .toLocaleLowerCase('en-US').match(/[\p{L}\p{N}]+/gu) ?? [];
  return qa ? tokens.filter(word => !['a', 'an', 'the'].includes(word)) : tokens;
}
/** Multiset token F1; repeated words count only when they occur on both sides. */
export function spanF1(actual: unknown, expected: unknown, qa = false): number {
  if (qa) {
    const numbers = (value: unknown) => String(value).match(/[+-]?\d+(?:\.\d+)?/g)?.map(Number) ?? [];
    const goldNumbers = numbers(expected), gotNumbers = numbers(actual);
    if (goldNumbers.length && canonical(goldNumbers.sort((a, b) => a - b)) !==
        canonical(gotNumbers.sort((a, b) => a - b))) return 0;
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
export type FilesOracle = { compare?: 'exact' | 'moves' | 'rewrite' | 'csv' | 'counts'; threshold?: number; span?: number;
  total?: number; rubric?: string; alternates?: Record<string, string[]>;
  /** Reports may only quote the corresponding original source. */
  quote_sources?: Record<string, string>;
  /** Alternate annotated clauses, keyed by the report row's id. */
  quote_alternates?: Record<string, string[]>;
  return_count?: 'changed' | 'csv_nonempty' | 'counts'; report?: string };
export type FilesVerdict = { accepted: boolean; score: number; passed: number; items: number; failed: string[];
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

export function checkFiles(actual: Record<string, string>, expected: Record<string, string>, input: Record<string, string>,
    spec: FilesOracle = {}, judgments: FilesVerdict['judgments'] = {}): FilesVerdict {
  const compare = spec.compare ?? 'exact', threshold = probability(spec.threshold ?? 0.9, 'files'),
    span = probability(spec.span ?? 0.5, 'files span');
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
  if (spec.return_count === 'changed') return Object.keys(files).filter(path => files[path] !== input[path]).length;
  const report = files[spec.report ?? '']; if (report === undefined) throw new Error('missing report');
  if (spec.return_count === 'counts') return countLines(report);
  return csvRows(report).slice(1).filter(row => row[1]?.trim()).length;
}
export function checkFileReturn(actual: unknown, files: Record<string, string>, input: Record<string, string>, spec: FilesOracle): boolean {
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
