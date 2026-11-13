import { checkConstraints, type WritingConstraint } from './constraints.js';
import { canonical, jsonStringRecordCanonical } from '../oracle-kit/json-record.js';
import { decimalForms } from '../oracle-kit/decimal.js';
import { assertKnownComparator, recordComparator } from '../oracle-kit/oracle-registry.js';
import { ANSWER_SET_NORMALIZATIONS, answerListEqual, numericClose, qaAliasMapScore, resultSetEqual } from './answer-sets.js';
import { PUZZLE_SOLUTION_NORMALIZATIONS, puzzleSolves } from './puzzle-solutions.js';
export { jsonStringRecordCanonical };
/** Answer checks used by generated and dataset-backed teacher cases. */
export type OracleLevel = 'exact' | 'normalized' | 'span' | 'agreement' | 'judged' | 'constraints';
export const ORACLE_LEVELS: readonly OracleLevel[] = ['exact', 'normalized', 'span', 'agreement', 'judged', 'constraints'];
/** Baseline answer comparison version; explicit specialized comparators carry their own verdict version. */
export const ANSWER_COMPARISON_VERSION = 'normalized-decimal-exact/4';
export type NumericListFieldSpec = { key: string; separator: '; '; empty_value: 'none' };
export type SemanticStructuredComparisonSpec = { schema_version: string; unordered_paths: string[]; prose_paths: string[];
  ordered_paths?: string[]; witness_review_required?: boolean; witness_policy?: string; prose_policy?: string };
export type OracleSpec = OracleLevel | { level: OracleLevel; alternates?: unknown[];
  threshold?: number; normalization?: 'qa' | 'qa-string-map' | 'named-tree' | 'json-string-record' | (string & {});
  comparison_spec?: SemanticStructuredComparisonSpec;
  /** Explicit answer-map keys whose string values are exact decimal quantities. */
  numeric_keys?: string[];
  /** Explicit answer-map keys whose values are ordered exact unsigned integer lists. */
  numeric_list_fields?: NumericListFieldSpec[];
  /** Explicit answer-map keys whose lowercase strings are unsigned numerals in a declared base. */
  radix_fields?: Array<{ key: string; base: number }>;
  rubric?: string; context?: unknown; [key: string]: unknown };
const BUILTIN_NORMALIZATIONS = ['qa', 'qa-string-map', 'named-tree', 'json-string-record', 'semantic-structured',
  ...ANSWER_SET_NORMALIZATIONS, ...PUZZLE_SOLUTION_NORMALIZATIONS] as const;
/** Apply only explicitly declared unordered array paths; object keys are already order-free. */
export function declaredStructuralCanonical(value: unknown, unorderedPaths: string[], prosePaths: string[] = [],
  ignoreProse = false): string | null {
  const parsePointer = (pointer: string): string[] => {
    if (!pointer.startsWith('/') || /~(?![01])/.test(pointer)) throw new RangeError(`invalid comparison path: ${pointer}`);
    return pointer.slice(1).split('/').map(part => part.replace(/~1/g, '/').replace(/~0/g, '~'));
  };
  const unordered = unorderedPaths.map(parsePointer), prose = prosePaths.map(parsePointer);
  const matches = (path: string[], pattern: string[]) => path.length === pattern.length &&
    path.every((part, index) => pattern[index] === '*' || pattern[index] === part);
  const visit = (item: unknown, path: string[]): string | null => {
    if (ignoreProse && prose.some(pattern => matches(path, pattern))) {
      if (typeof item !== 'string') return null;
      return JSON.stringify('__reviewed_prose__');
    }
    if (Array.isArray(item)) {
      const parts = item.map((child, index) => visit(child, [...path, String(index)]));
      if (parts.some(part => part === null)) return null;
      if (unordered.some(pattern => matches(path, pattern))) parts.sort();
      return `[${parts.join(',')}]`;
    }
    if (item && typeof item === 'object') {
      const fields = Object.entries(item as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
      const parts = fields.map(([key, child]) => [key, visit(child, [...path, key])] as const);
      if (parts.some(([, child]) => child === null)) return null;
      return `{${parts.map(([key, child]) => `${JSON.stringify(key)}:${child}`).join(',')}}`;
    }
    if (item === null || typeof item === 'string' || typeof item === 'boolean' ||
        (typeof item === 'number' && Number.isFinite(item))) return JSON.stringify(item);
    return null;
  };
  return visit(value, []);
}
const BUILTIN_FILE_COMPARES = ['content', 'exact', 'moves', 'rewrite', 'csv', 'counts', 'json-string-record', 'qa-string-map',
  'json-semantic-structured', 'ordinal-distance-json-record',
  'markdown-terminal-newline'] as const;
export type OracleVerdict = { accepted: boolean; level: OracleLevel; score?: number; verdict?: string;
  needs_review?: boolean; comparison_version?: string };

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

/** JSON string maps keep exact values unless a source contract explicitly marks decimal-valued keys. */
export function jsonStringRecordWithNumericKeysCanonical(value: unknown, numericKeys: unknown = []): string | null {
  if (!Array.isArray(numericKeys) || numericKeys.some(key => typeof key !== 'string') ||
      new Set(numericKeys).size !== numericKeys.length) return null;
  let record: Record<string, unknown>;
  if (typeof value === 'string') {
    if (jsonStringRecordCanonical(value) === null) return null;
    try { record = JSON.parse(value) as Record<string, unknown>; } catch { return null; }
  } else if (value && typeof value === 'object' && !Array.isArray(value)) {
    try {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) return null;
      record = Object.create(null) as Record<string, unknown>;
      for (const key of Object.keys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor?.enumerable || !('value' in descriptor) || typeof descriptor.value !== 'string') return null;
        record[key] = descriptor.value;
      }
    } catch { return null; }
  } else return null;

  for (const key of numericKeys as string[]) {
    if (!Object.hasOwn(record, key) || typeof record[key] !== 'string' || decimalForms(record[key] as string) === null)
      return null;
  }
  const selected = new Set(numericKeys as string[]);
  return JSON.stringify(Object.entries(record).sort(([a], [b]) => a.localeCompare(b)).map(([key, entry]) => {
    if (!selected.has(key)) return [key, { exact_string: entry }];
    return [key, { exact_decimal: decimalForms(entry as string)!.exact }];
  }));
}

/** Canonicalize only explicitly declared unsigned radix fields to exact BigInt values.
 * Digits are lowercase 0-9a-z, bases 2..36, and leading zeroes are equivalent.
 * No signs, uppercase, whitespace, floats, or undeclared-field coercion are accepted.
 */
export function jsonStringRecordWithRadixFieldsCanonical(value: unknown, radixFields: unknown = []): string | null {
  if (!Array.isArray(radixFields)) return null;
  const fields = new Map<string, number>();
  for (const field of radixFields) {
    if (!field || typeof field !== 'object' || typeof field.key !== 'string' ||
        !Number.isInteger(field.base) || field.base < 2 || field.base > 36 || fields.has(field.key)) return null;
    fields.set(field.key, field.base);
  }
  let record: Record<string, unknown>;
  if (typeof value === 'string') {
    if (jsonStringRecordCanonical(value) === null) return null;
    try { record = JSON.parse(value) as Record<string, unknown>; } catch { return null; }
  } else if (value && typeof value === 'object' && !Array.isArray(value)) {
    try {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) return null;
      record = Object.create(null) as Record<string, unknown>;
      for (const key of Object.keys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor?.enumerable || !('value' in descriptor) || typeof descriptor.value !== 'string') return null;
        record[key] = descriptor.value;
      }
    } catch { return null; }
  } else return null;
  for (const key of fields.keys()) if (!Object.hasOwn(record, key) || typeof record[key] !== 'string') return null;
  const parsed = new Map<string, bigint>();
  for (const [key, base] of fields) {
    const digits = record[key] as string;
    if (!digits || !/^[0-9a-z]+$/.test(digits)) return null;
    let number = 0n;
    for (const character of digits) {
      const digit = '0123456789abcdefghijklmnopqrstuvwxyz'.indexOf(character);
      if (digit < 0 || digit >= base) return null;
      number = number * BigInt(base) + BigInt(digit);
    }
    parsed.set(key, number);
  }
  return JSON.stringify(Object.entries(record).sort(([a], [b]) => a.localeCompare(b)).map(([key, entry]) =>
    fields.has(key) ? [key, { exact_radix_integer: parsed.get(key)!.toString() }] : [key, { exact_string: entry }]));
}

/** Canonicalize explicitly declared measured-count lists only. Grouping commas are accepted
 * only in standard three-digit groups; order and list membership remain exact. */
export function jsonStringRecordWithNumericListFieldsCanonical(value: unknown, numericListFields: unknown = []): string | null {
  if (!validNumericListFieldSchema(numericListFields)) return null;
  const fields = new Map<string, NumericListFieldSpec>();
  for (const field of numericListFields) {
    fields.set(field.key, field as NumericListFieldSpec);
  }
  let record: Record<string, unknown>;
  if (typeof value === 'string') {
    if (jsonStringRecordCanonical(value) === null) return null;
    try { record = JSON.parse(value) as Record<string, unknown>; } catch { return null; }
  } else if (value && typeof value === 'object' && !Array.isArray(value)) {
    try {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) return null;
      record = Object.create(null) as Record<string, unknown>;
      for (const key of Object.keys(value)) {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor?.enumerable || !('value' in descriptor) || typeof descriptor.value !== 'string') return null;
        record[key] = descriptor.value;
      }
    } catch { return null; }
  } else return null;

  const parsed = new Map<string, string[]>();
  for (const [key, field] of fields) {
    if (!Object.hasOwn(record, key) || typeof record[key] !== 'string') return null;
    const text = record[key] as string;
    if (text === field.empty_value) { parsed.set(key, []); continue; }
    const items = text.split(field.separator);
    if (!items.length) return null;
    const canonicalItems: string[] = [];
    for (const item of items) {
      if (!/^(?:\d+|[1-9]\d{0,2}(?:,\d{3})+)$/.test(item)) return null;
      canonicalItems.push(BigInt(item.replaceAll(',', '')).toString());
    }
    parsed.set(key, canonicalItems);
  }
  return JSON.stringify(Object.entries(record).sort(([a], [b]) => a.localeCompare(b)).map(([key, entry]) =>
    fields.has(key) ? [key, { exact_unsigned_integer_list: parsed.get(key)! }] : [key, { exact_string: entry }]));
}

function validNumericListFieldSchema(value: unknown): value is NumericListFieldSpec[] {
  if (!Array.isArray(value)) return false;
  const keys = new Set<string>();
  for (const field of value) {
    if (!field || typeof field !== 'object' || Array.isArray(field) ||
        Object.keys(field).sort().join(',') !== 'empty_value,key,separator' ||
        typeof field.key !== 'string' || !field.key || field.separator !== '; ' || field.empty_value !== 'none' ||
        keys.has(field.key)) return false;
    keys.add(field.key);
  }
  return true;
}

function jsonStringRecordAnswerCanonical(value: unknown, numericKeys: unknown, radixFields: unknown, numericListFields: unknown): string | null {
  if (radixFields !== undefined && !Array.isArray(radixFields)) return null;
  if (numericListFields !== undefined && !validNumericListFieldSchema(numericListFields)) return null;
  const declared = [numericKeys, radixFields, numericListFields].filter(value => Array.isArray(value) && value.length > 0).length;
  if (declared > 1) return null;
  if (Array.isArray(numericListFields) && numericListFields.length)
    return jsonStringRecordWithNumericListFieldsCanonical(value, numericListFields);
  return Array.isArray(radixFields) && radixFields.length ?
    jsonStringRecordWithRadixFieldsCanonical(value, radixFields) : jsonStringRecordWithNumericKeysCanonical(value, numericKeys);
}

function normalizeText(value: unknown): string {
  const text = String(value ?? '').normalize('NFKC').trim().toLocaleLowerCase('en-US').replace(/\s+/g, ' ');
  const number = text.replace(/(?<=\d),(?=\d{3}(?:\D|$))/g, '');
  if (number.length <= 512 && /^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(number)) {
    // Reuse only TaTQA's exact decimal representation; generic normalized answers must never
    // inherit its optional source-specific two-decimal rounding equivalence.
    const exact = decimalForms(number)?.exact ?? (number.endsWith('.') ? decimalForms(`${number}0`)?.exact : undefined);
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
function qaNumbers(value: unknown): string[] {
  return String(value).match(/[+-]?\d+(?:\.\d+)?/g)?.map(token =>
    decimalForms(token)?.exact ?? token).sort() ?? [];
}
/** A yes/no answer's polarity: `true`, `"yes"` and `"true"` are true, `false`, `"no"` and `"false"` false, else undefined. */
function polarity(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value;
  if (typeof value !== 'string') return undefined;
  const text = normalizeText(value).replace(/[.!]$/, '');
  return text === 'yes' || text === 'true' ? true : text === 'no' || text === 'false' ? false : undefined;
}
/** A string answer written as a JSON string literal (`"\"#B A#\""`) is that string when the gold is a string that is not
 * itself quoted. */
function unquotedAnswer(actual: unknown, candidates: unknown[]): unknown {
  if (typeof actual !== 'string') return actual;
  const text = actual.trim();
  if (text.length < 2 || !text.startsWith('"') || !text.endsWith('"')) return actual;
  if (!candidates.some(candidate => typeof candidate === 'string') ||
      candidates.some(candidate => typeof candidate === 'string' && candidate.trim().startsWith('"'))) return actual;
  try {
    const inner: unknown = JSON.parse(text);
    return typeof inner === 'string' ? inner : actual;
  } catch { return actual; }
}
/**
 * Extractive QA: a short answer that holds the whole gold span of two or more tokens, contiguous and in order, with the
 * same numbers, and is at most twice the gold's length ("Celine Dion's \"Because You Loved Me\"" for "Because You Loved
 * Me"). One-token golds stay with F1: "Duke Lemur Center" must not pass for "Duke".
 */
export function containsQaSpan(actual: unknown, expected: unknown): boolean {
  if (canonical(qaNumbers(actual)) !== canonical(qaNumbers(expected))) return false;
  const got = words(actual, true), want = words(expected, true);
  if (want.length < 2 || got.length > 2 * want.length) return false;
  for (let start = 0; start + want.length <= got.length; start++)
    if (want.every((word, index) => got[start + index] === word)) return true;
  return false;
}
const QA_HEDGES = new Set(['or', 'and', 'nor', 'either', 'neither', 'not', 'no', 'never', 'maybe', 'perhaps', 'possibly',
  'probably', 'unclear', 'unknown']);
const QA_POSSESSOR = /^\s*(?:(?:the|a|an)\s+)?(?:(?:my|your|his|her|its|our|their)\s+|(?:[\p{L}\p{N}][\p{L}\p{N}.-]*\s+)?[\p{L}\p{N}][\p{L}\p{N}.-]*['\u2019]s\s+)(?=\S)/iu;
/**
 * Extractive QA: the core of a short answer that restates part of the question or names a possessor before the answer
 * ("just had sex" for "What did A say they just had ...?", "the alien's spaceship" for "the spaceship"). Drops a leading
 * possessor (`X's`, `X Y's`, a possessive determiner), then the longest leading run of words that occurs contiguously in
 * the question and holds no gold word. A core that is empty, unchanged, or hedges (or/and/not/maybe ... absent from the
 * gold: "sex or coffee") is not offered, so a multi-answer reply never passes this way.
 */
export function qaAnswerCores(actual: unknown, expected: unknown, question?: string): string[] {
  if (typeof actual !== 'string') return [];
  const gold = new Set(words(expected, true));
  const cores = new Set<string>();
  const echoFree = (tokens: string[]): string[] => {
    const asked = question ? words(question, true) : [];
    for (let k = tokens.length - 1; k >= 1; k--) {
      const lead = tokens.slice(0, k);
      if (lead.some(word => gold.has(word))) continue;
      for (let start = 0; start + k <= asked.length; start++)
        if (lead.every((word, index) => asked[start + index] === word)) return tokens.slice(k);
    }
    return tokens;
  };
  const possessorFree = actual.replace(QA_POSSESSOR, '');
  for (const text of new Set([actual, possessorFree])) {
    const core = echoFree(words(text, true));
    if (core.length) cores.add(core.join(' '));
  }
  cores.delete(words(actual, true).join(' '));
  return [...cores].filter(core => !core.split(' ').some(word => QA_HEDGES.has(word) && !gold.has(word)));
}
/** Multiset token F1; repeated words count only when they occur on both sides. */
export function spanF1(actual: unknown, expected: unknown, qa = false): number {
  if (qa) {
    const goldNumbers = qaNumbers(expected), gotNumbers = qaNumbers(actual);
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
export const DATA_QUALITY_VERSION = 3;
export const FILE_CONTENT_COMPARISON_VERSION = 'json-content/3';
export type FilesOracle = { compare?: 'content' | 'exact' | 'moves' | 'rewrite' | 'csv' | 'counts' | 'json-string-record' | 'qa-string-map' | 'json-semantic-structured' | 'markdown-terminal-newline' | (string & {}); threshold?: number; span?: number;
  total?: number; rubric?: string; alternates?: Record<string, string[]>;
  /** JSON outputs whose prose-only fields may vary while all other structure remains exact. */
  json_report_paths?: string[];
  comparison_spec?: SemanticStructuredComparisonSpec;
  /** Source files bound into the prose judge context for JSON reports. */
  rationale_context_paths?: string[];
  /** Explicit answer file for return-value consistency when the default answers.json names do not apply. */
  return_path?: string;
  /** `content` compares: whether one terminal LF/CRLF of a non-JSON file counts (`exact`, the default) or not
   * (`lenient`: the collector's choice when the task's instructions never mention a newline). */
  terminal_newline?: 'exact' | 'lenient';
  /** Explicit answer-map keys whose string values are exact decimal quantities. */
  numeric_keys?: string[];
  /** Explicit answer-map keys whose values are ordered exact unsigned integer lists. */
  numeric_list_fields?: NumericListFieldSpec[];
  /** Explicit answer-map fields whose lowercase values are unsigned numerals in a declared base. */
  radix_fields?: Array<{ key: string; base: number }>;
  /** Exact output-map label domain for bounded ordinal source references. */
  ordinal_levels?: string[];
  /** Maximum per-entry ordinal distance for the explicitly selected output map. */
  ordinal_distance_max?: number;
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

function semanticJsonSpecValid(spec: FilesOracle): boolean {
  const paths = spec.json_report_paths, context = spec.rationale_context_paths, comparison = spec.comparison_spec;
  if (!Array.isArray(paths) || !paths.length || new Set(paths).size !== paths.length ||
      paths.some(path => typeof path !== 'string' || !path || path.startsWith('/') || path.includes('\\') ||
        path.split('/').some(part => !part || part === '.' || part === '..') || !/\.json$/i.test(path)) ||
      !Array.isArray(context) || !context.length || new Set(context).size !== context.length ||
      context.some(path => typeof path !== 'string' || !path || path.startsWith('/') || path.includes('\\') ||
        path.split('/').some(part => !part || part === '.' || part === '..')) ||
      !comparison || comparison.schema_version !== 'semantic-structured-comparison/1' ||
      !Array.isArray(comparison.unordered_paths) || !Array.isArray(comparison.ordered_paths) ||
      !Array.isArray(comparison.prose_paths) || !comparison.prose_paths.length ||
      typeof comparison.prose_policy !== 'string' || !comparison.prose_policy.trim()) return false;
  const declared = [...comparison.unordered_paths, ...comparison.ordered_paths, ...comparison.prose_paths];
  return declared.every(path => typeof path === 'string' && path.startsWith('/') && !/~(?![01])/.test(path)) &&
    new Set(declared).size === declared.length;
}

function hasRequiredJsonTextFields(value: unknown, pointers: string[]): boolean {
  const read = (current: unknown, parts: string[], index: number): boolean => {
    if (index === parts.length) return typeof current === 'string' && !!current.trim();
    const part = parts[index]!;
    if (part === '*') return Array.isArray(current) && current.length > 0 &&
      current.every(child => read(child, parts, index + 1));
    if (!current || typeof current !== 'object' || Array.isArray(current) ||
        !Object.hasOwn(current, part)) return false;
    return read((current as Record<string, unknown>)[part], parts, index + 1);
  };
  return pointers.every(pointer => read(value, pointer.slice(1).split('/').map(part =>
    part.replace(/~1/g, '/').replace(/~0/g, '~')), 0));
}

function semanticJsonFileComparison(actual: Record<string, string>, expected: Record<string, string>,
  input: Record<string, string>, spec: FilesOracle): FilesVerdict {
  const invalid = !semanticJsonSpecValid(spec);
  const configured = new Set(spec.json_report_paths ?? []);
  const paths = [...new Set([...Object.keys(expected), ...Object.keys(actual), ...Object.keys(input)])].sort();
  const failed: string[] = [], errors: string[] = [], pending: string[] = [];
  if (invalid) errors.push('invalid_json_semantic_structured_spec');
  if (!invalid) {
    for (const path of spec.rationale_context_paths!) if (typeof input[path] !== 'string')
      errors.push(`missing_rationale_context:${path}`);
    for (const path of configured) if (typeof expected[path] !== 'string' || typeof input[path] === 'string')
      errors.push(`invalid_json_report_path:${path}`);
  }
  for (const path of paths) {
    if (configured.has(path)) {
      const gotText = actual[path], wantText = expected[path];
      if (typeof gotText !== 'string' || typeof wantText !== 'string' ||
          jsonFileCanonical(gotText) === null || jsonFileCanonical(wantText) === null) {
        failed.push(path); continue;
      }
      let got: unknown, want: unknown;
      try { got = JSON.parse(gotText); want = JSON.parse(wantText); } catch { failed.push(path); continue; }
      const comparison = spec.comparison_spec!;
      if (!hasRequiredJsonTextFields(got, comparison.prose_paths) ||
          !hasRequiredJsonTextFields(want, comparison.prose_paths)) { failed.push(path); continue; }
      const strictGot = declaredStructuralCanonical(got, comparison.unordered_paths);
      const strictWant = declaredStructuralCanonical(want, comparison.unordered_paths);
      if (strictGot === null || strictWant === null) { failed.push(path); continue; }
      if (strictGot !== strictWant) {
        const proseGot = declaredStructuralCanonical(got, comparison.unordered_paths, comparison.prose_paths, true);
        const proseWant = declaredStructuralCanonical(want, comparison.unordered_paths, comparison.prose_paths, true);
        if (proseGot === null || proseWant === null || proseGot !== proseWant) failed.push(path);
      }
      if (!failed.includes(path)) pending.push(`unjudged_json_rationale:${path}`);
      continue;
    }
    if (!fileContentEqual(path, actual[path], expected[path])) failed.push(path);
    if (typeof expected[path] === 'string' && expected[path] === input[path] && actual[path] !== expected[path])
      errors.push(`unexpected_change:${path}`);
  }
  const items = paths.length, passed = items - failed.length;
  return { accepted: !invalid && !errors.length && !failed.length && !pending.length,
    score: items ? passed / items : 1, passed, items, failed: failed.slice(0, 20), errors, pending,
    quality_version: DATA_QUALITY_VERSION, comparison_version: 'json-semantic-structured-files/1' };
}

/** Remove at most one terminal LF or CRLF. No other whitespace or line-ending normalization occurs. */
export function markdownTerminalNewlineBody(text: string): string {
  return text.endsWith('\r\n') ? text.slice(0, -2) : text.endsWith('\n') ? text.slice(0, -1) : text;
}
export function markdownTerminalNewlineEqual(path: string, actual: string | undefined, expected: string | undefined): boolean {
  return typeof actual === 'string' && typeof expected === 'string' && /\.md$/i.test(path) &&
    markdownTerminalNewlineBody(actual) === markdownTerminalNewlineBody(expected);
}

/** Compare a source-bound map of ordinal labels without relaxing its keys or label domain. */
export function ordinalDistanceRecordReview(actual: unknown, expected: unknown, levels: unknown, maxDistance: unknown) {
  const invalid = (error: string) => ({ accepted: false, errors: [error], entries: [] as Array<Record<string, unknown>> });
  if (!Array.isArray(levels) || levels.length < 2 || levels.some(level => typeof level !== 'string' || !level) ||
      new Set(levels).size !== levels.length || !Number.isSafeInteger(maxDistance) || Number(maxDistance) < 0)
    return invalid('invalid_ordinal_record_spec');
  const parseRecord = (value: unknown): Record<string, unknown> | null => {
    if (typeof value === 'string') { try { value = JSON.parse(value); } catch { return null; } }
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return null;
    return value as Record<string, unknown>;
  };
  const got = parseRecord(actual), want = parseRecord(expected);
  if (!got || !want) return invalid('invalid_ordinal_record_value');
  const gotKeys = Object.keys(got).sort(), wantKeys = Object.keys(want).sort();
  if (canonical(gotKeys) !== canonical(wantKeys)) return invalid('ordinal_record_keys_mismatch');
  const order = new Map((levels as string[]).map((label, index) => [label, index]));
  const entries = gotKeys.map(key => {
    const actualLabel = got[key], expectedLabel = want[key];
    const actualIndex = typeof actualLabel === 'string' ? order.get(actualLabel) : undefined;
    const expectedIndex = typeof expectedLabel === 'string' ? order.get(expectedLabel) : undefined;
    const distance = actualIndex === undefined || expectedIndex === undefined ? null : Math.abs(actualIndex - expectedIndex);
    return { key, actual: actualLabel, expected: expectedLabel, ordinal_distance: distance,
      accepted: distance !== null && distance <= Number(maxDistance) };
  });
  const errors = entries.filter(entry => !entry.accepted).map(entry =>
    entry.ordinal_distance === null ? `unknown_ordinal_label:${entry.key}` : `ordinal_distance_exceeded:${entry.key}`);
  return { accepted: errors.length === 0, errors, entries };
}

export function checkFiles(actual: Record<string, string>, expected: Record<string, string>, input: Record<string, string>,
    spec: FilesOracle = {}, judgments: FilesVerdict['judgments'] = {}): FilesVerdict {
  assertKnownComparator('compare', spec.compare, BUILTIN_FILE_COMPARES);
  if (spec.return_path !== undefined && (typeof spec.return_path !== 'string' || !spec.return_path || spec.return_path.startsWith('/') ||
      spec.return_path.includes('\\') || spec.return_path.split('/').some(part => !part || part === '.' || part === '..')))
    return { accepted: false, score: 0, passed: 0, items: 0, failed: [], errors: ['invalid_return_path'], pending: [], quality_version: DATA_QUALITY_VERSION };
  if (spec.compare === 'json-string-record' && spec.numeric_list_fields !== undefined &&
      !validNumericListFieldSchema(spec.numeric_list_fields))
    return { accepted: false, score: 0, passed: 0, items: 0, failed: [], errors: ['invalid_numeric_list_field_schema'], pending: [], quality_version: DATA_QUALITY_VERSION };
  if (spec.compare === 'json-semantic-structured')
    return semanticJsonFileComparison(actual, expected, input, spec);
  const compare = spec.compare ?? 'exact', threshold = probability(spec.threshold ?? 0.9, 'files'),
    span = probability(spec.span ?? 0.5, 'files span');
  if (compare === 'content') {
    const paths = [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort();
    const lenient = spec.terminal_newline === 'lenient';
    const failed = paths.filter(path => !fileContentEqual(path, actual[path], expected[path]) &&
      !(lenient && !/\.json$/i.test(path) && typeof actual[path] === 'string' && typeof expected[path] === 'string' &&
        markdownTerminalNewlineBody(actual[path]!) === markdownTerminalNewlineBody(expected[path]!)));
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
    const got = actual[path], want = expected[path], before = input[path];
    // Read-only source corruption and unexpected output are contract failures, never diluted by many report rows.
    if (want === input[path] && got !== want || want === undefined && got !== undefined) errors.push(`unexpected_change:${path}`);
    if (got === undefined || want === undefined) { item(path, got === want); continue; }
    // Existing source files are checked exactly, even in a CSV/rewrite task.
    if (want === input[path]) { item(path, got === want); continue; }
    if (compare === 'ordinal-distance-json-record') {
      const review = ordinalDistanceRecordReview(got, want, spec.ordinal_levels, spec.ordinal_distance_max);
      item(path, path === 'decisions.json' && review.accepted);
      if (!review.accepted) errors.push(...review.errors.map(error => `${path}:${error}`));
      continue;
    }
    if (compare === 'json-string-record') {
      const parsed = jsonStringRecordAnswerCanonical(got, spec.numeric_keys, spec.radix_fields, spec.numeric_list_fields);
      const candidates = [want, ...(spec.alternates?.[path] ?? [])];
      item(path, parsed !== null && candidates.some(candidate =>
        parsed === jsonStringRecordAnswerCanonical(candidate, spec.numeric_keys, spec.radix_fields, spec.numeric_list_fields)));
      continue;
    }
    if (compare === 'qa-string-map') {
      const parsed = qaStringMapCanonical(got);
      item(path, path === 'answers.json' && parsed !== null && parsed === qaStringMapCanonical(want));
      continue;
    }
    const registered = recordComparator(compare);
    if (registered) {
      item(path, path === registered.answerFile && registered.equal(got, want));
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
    } else if (compare === 'rewrite' && typeof want === 'string') {
      // A rewrite may create a new expected file. In that case there is no
      // input front matter to preserve, but the candidate must still match
      // the reference's front matter and be judged against the exact output
      // path. Unexpected extra files (want === undefined) remain exact fails.
      const validShape = got !== undefined && front(got) === front(want) &&
        (before === undefined || front(got) === front(before));
      const referenceMatch = got !== undefined && [want, ...(spec.alternates?.[path] ?? [])]
        .some(candidate => body(got).trim() === body(candidate).trim());
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
    ...(compare === 'json-string-record' ? { comparison_version: spec.numeric_list_fields?.length ?
      'json-string-record-integer-list-fields/1' : spec.numeric_keys?.length ?
      'json-string-record-decimal-fields/1' : spec.radix_fields?.length ? 'json-string-record-radix-fields/1' :
        'json-string-record-object/1' } : {}),
    ...(compare === 'csv' ? { positive_recall: positiveRecall, positive_precision: positivePrecision } : {}),
    ...(Object.keys(judgments).length ? { judgments } : {}) };
}

/** Non-reference rewrites require a separately configured judge; a word-overlap heuristic cannot certify them. */
export async function checkFilesWithJudge(actual: Record<string, string>, expected: Record<string, string>,
    input: Record<string, string>, spec: FilesOracle = {}, judge?: OracleJudge): Promise<FilesVerdict> {
  const first = checkFiles(actual, expected, input, spec);
  if (spec.compare === 'json-semantic-structured') {
    if (first.errors.length || first.failed.length || !semanticJsonSpecValid(spec)) return first;
    const pending: string[] = [], failed = [...first.failed], judgments: NonNullable<FilesVerdict['judgments']> = {};
    for (const path of spec.json_report_paths!) {
      let got: unknown, want: unknown;
      try { got = JSON.parse(actual[path]!); want = JSON.parse(expected[path]!); }
      catch { failed.push(path); continue; }
      if (!judge) { pending.push(`unjudged_json_rationale:${path}`); continue; }
      const sourceFiles = Object.fromEntries(spec.rationale_context_paths!.map(sourcePath => [sourcePath, input[sourcePath]]));
      const verdict = await judge({ actual: { output_path: path, report: got, source_files: sourceFiles },
        expected: { checked_non_prose_structure: true, prose_paths: spec.comparison_spec!.prose_paths },
        rubric: spec.comparison_spec!.prose_policy! });
      judgments[path] = { accepted: verdict.accepted === true, verdict: verdict.verdict,
        ...(verdict.needs_review ? { needs_review: true } : {}) };
      if (verdict.accepted !== true || verdict.needs_review) {
        if (verdict.needs_review) pending.push(`unverified_json_rationale:${path}`);
        else failed.push(path);
      }
    }
    return { ...first, accepted: !failed.length && !pending.length,
      score: first.items ? (first.items - failed.length) / first.items : 1,
      failed: failed.slice(0, 20), pending, judgments: Object.keys(judgments).length ? judgments : undefined,
      comparison_version: 'json-semantic-structured-files/1' };
  }
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
    if (actual[path] === undefined || expected[path] === input[path] ||
        (input[path] !== undefined && front(actual[path]!) !== front(input[path]!)) ||
        front(actual[path]!) !== front(expected[path]!)) continue;
    if ([expected[path]!, ...(spec.alternates?.[path] ?? [])].some(value => body(actual[path]!).trim() === body(value).trim())) continue;
    judgments[path] = await judge({ actual: {
      original_exists: input[path] !== undefined,
      original: input[path] === undefined ? null : body(input[path]),
      revised: body(actual[path]!),
    }, expected: { reference: body(expected[path]!) }, rubric: spec.rubric });
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
  const registered = recordComparator(spec.compare);
  if (registered) {
    const written = files[registered.answerFile];
    return written !== undefined && registered.equal(actual, written);
  }
  if (spec.compare === 'json-string-record') {
    if (spec.return_path !== undefined && (typeof spec.return_path !== 'string' || !spec.return_path || spec.return_path.startsWith('/') ||
        spec.return_path.includes('\\') || spec.return_path.split('/').some(part => !part || part === '.' || part === '..'))) return false;
    if (spec.numeric_list_fields !== undefined && !validNumericListFieldSchema(spec.numeric_list_fields)) return false;
    const answerPath = spec.return_path;
    const answerFiles = answerPath === undefined ? ['answers.json', 'answer.json'].filter(path => Object.hasOwn(files, path)) : [answerPath];
    if (answerFiles.length !== 1 || !Object.hasOwn(files, answerFiles[0]!)) return false;
    const written = files[answerFiles[0]!];
    const returned = jsonStringRecordAnswerCanonical(actual, spec.numeric_keys, spec.radix_fields, spec.numeric_list_fields);
    return written !== undefined && returned !== null &&
      returned === jsonStringRecordAnswerCanonical(written, spec.numeric_keys, spec.radix_fields, spec.numeric_list_fields);
  }
  if (!spec.return_count) return true;
  try { return canonical(actual) === canonical(fileReturnValue(files, input, spec)); } catch { return false; }
}

/** A judge's verdict is supplied by the collection caller; no case can self-certify its own prose. */
export async function checkOracle(actual: unknown, expected: unknown, oracle: OracleSpec = 'exact',
  judge?: (input: { actual: unknown; expected: unknown; rubric: string }) => Promise<{ accepted: boolean; verdict: string; needs_review?: boolean }>,
  /** The task's question (its `question` input), for extractive QA answers that restate it (`qaAnswerCores`). */
  context: { question?: string } = {}):
  Promise<OracleVerdict> {
  const spec = typeof oracle === 'string' ? { level: oracle } : oracle;
  const { level } = spec;
  if (!ORACLE_LEVELS.includes(level)) throw new RangeError(`unknown oracle level: ${level}`);
  const candidates = [expected, ...('alternates' in spec ? spec.alternates ?? [] : [])];
  actual = unquotedAnswer(actual, candidates);
  if (level === 'exact') return { accepted: candidates.some(candidate => canonical(actual) === canonical(candidate)), level };
  if (level === 'normalized' && spec.normalization === 'ordinal-distance-record') {
    const review = ordinalDistanceRecordReview(actual, expected, spec.ordinal_levels, spec.ordinal_distance_max);
    return { accepted: review.accepted, level, verdict: JSON.stringify(review),
      comparison_version: 'bounded-ordinal-record/1' };
  }
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
    if (spec.numeric_list_fields !== undefined && !validNumericListFieldSchema(spec.numeric_list_fields))
      return { accepted: false, level, comparison_version: 'invalid-json-string-record-spec/1' };
    const answer = jsonStringRecordAnswerCanonical(actual, spec.numeric_keys, spec.radix_fields, spec.numeric_list_fields);
    return { accepted: answer !== null && candidates.some(candidate =>
      jsonStringRecordAnswerCanonical(candidate, spec.numeric_keys, spec.radix_fields, spec.numeric_list_fields) === answer), level,
      comparison_version: spec.numeric_list_fields?.length ? 'json-string-record-integer-list-fields/1' : spec.numeric_keys?.length ?
        'json-string-record-decimal-fields/1' : spec.radix_fields?.length ?
        'json-string-record-radix-fields/1' : 'json-string-record-object/1' };
  }
  if (level === 'normalized' && spec.normalization === 'semantic-structured') {
    const comparison = spec.comparison_spec;
    const allPaths = comparison ? [...comparison.unordered_paths ?? [], ...comparison.ordered_paths ?? [],
      ...comparison.prose_paths ?? []] : [];
    if (comparison?.schema_version !== 'semantic-structured-comparison/1' ||
        !Array.isArray(comparison.unordered_paths) || !Array.isArray(comparison.ordered_paths) ||
        !Array.isArray(comparison.prose_paths) || new Set(allPaths).size !== allPaths.length ||
        allPaths.some(path => typeof path !== 'string' || !path.startsWith('/') || /~(?![01])/.test(path)))
      return { accepted: false, level, comparison_version: 'invalid-semantic-structured-spec/1' };
    const strict = declaredStructuralCanonical(actual, comparison.unordered_paths);
    const references = candidates.map(candidate => declaredStructuralCanonical(candidate, comparison.unordered_paths));
    if (strict !== null && references.includes(strict))
      return { accepted: true, level, comparison_version: 'declared-semantic-structure/1' };
    const withoutProse = declaredStructuralCanonical(actual, comparison.unordered_paths, comparison.prose_paths, true);
    const proseEquivalentStructure = withoutProse !== null && candidates.some(candidate =>
      declaredStructuralCanonical(candidate, comparison.unordered_paths, comparison.prose_paths, true) === withoutProse);
    if (proseEquivalentStructure || comparison.witness_review_required) {
      // A witness alternative may change IDs, quantities, or route structure.
      // A prose judge cannot waive those exact bindings. Such outputs need a
      // separate source-grounded witness review, not this generic comparator.
      if (proseEquivalentStructure && judge) {
        const verdict = await judge({ actual, expected: { answer: expected, context: spec.context },
          rubric: [comparison.witness_policy, comparison.prose_policy].filter(Boolean).join(' ') });
        return { accepted: verdict.accepted === true && !verdict.needs_review, level,
          verdict: verdict.verdict, ...(verdict.needs_review ? { needs_review: true } : {}),
          comparison_version: 'declared-semantic-structure/1' };
      }
      return { accepted: false, level, needs_review: true,
        verdict: 'structural alternative or prose equivalence needs independent source-grounded review',
        comparison_version: 'declared-semantic-structure/1' };
    }
    return { accepted: false, level, comparison_version: 'declared-semantic-structure/1' };
  }
  if (level === 'normalized' && spec.normalization === 'result-set')
    return { accepted: candidates.some(candidate => resultSetEqual(actual, candidate, spec.ordered === true)), level,
      comparison_version: 'result-set/1' };
  if (level === 'normalized' && spec.normalization === 'numeric-tolerance') {
    const options = { tolerance: typeof spec.tolerance === 'number' ? spec.tolerance : undefined,
      abs_tolerance: typeof spec.abs_tolerance === 'number' ? spec.abs_tolerance : undefined,
      percent_equivalent: spec.percent_equivalent === true };
    return { accepted: candidates.some(candidate => numericClose(actual, candidate, options)), level,
      comparison_version: 'numeric-tolerance/1' };
  }
  // `expected` is the puzzle instance; any answer that solves it is accepted (evaluation/puzzle-solutions.ts).
  if (level === 'normalized' && spec.normalization === 'puzzle-solution')
    return { accepted: puzzleSolves(actual, expected, spec.puzzle), level, comparison_version: 'puzzle-solution/1' };
  if (level === 'normalized' && spec.normalization === 'answer-list')
    return { accepted: candidates.some(candidate => answerListEqual(actual, candidate)), level, comparison_version: 'answer-list/1' };
  if (level === 'normalized' && spec.normalization === 'qa-alias-map') {
    const threshold = typeof spec.threshold === 'number' ? probability(spec.threshold, 'qa-alias-map') : 0.8;
    const results = candidates.map(candidate => qaAliasMapScore(actual, candidate, (got, gold) => spanF1(got, gold, true), threshold));
    const best = results.find(result => result.accepted) ?? results[0]!;
    return { accepted: best.accepted, level, comparison_version: 'qa-alias-map/1',
      ...(best.failed.length ? { verdict: `keys below threshold: ${best.failed.join(', ')}` } : {}) };
  }
  const registered = level === 'normalized' ? recordComparator(spec.normalization) : undefined;
  if (level === 'normalized' && spec.normalization === 'qa') {
    // Explicit scalar QA normalization was advertised but previously fell
    // through to generic text equality, rejecting articles and punctuation.
    const same = (candidate: unknown) => typeof actual === 'string' && typeof candidate === 'string' &&
      qaAnswerText(candidate).length > 0 && qaAnswerText(actual) === qaAnswerText(candidate) &&
      canonical(qaNumbers(actual)) === canonical(qaNumbers(candidate));
    const samePolarity = (candidate: unknown) => polarity(actual) !== undefined && polarity(actual) === polarity(candidate);
    return { accepted: candidates.some(candidate => same(candidate) || samePolarity(candidate)), level,
      comparison_version: 'qa-scalar-exact/2' };
  }
  if (registered) return { accepted: candidates.some(candidate => registered.equal(actual, candidate)), level };
  if (level === 'normalized') assertKnownComparator('normalization', spec.normalization, BUILTIN_NORMALIZATIONS);
  if (level === 'normalized') return { accepted: candidates.some(candidate => normalized(actual) === normalized(candidate) ||
    polarity(actual) !== undefined && polarity(actual) === polarity(candidate)), level };
  if (level === 'agreement') {
    const threshold = probability('threshold' in spec ? spec.threshold ?? 0.9 : 0.9, 'agreement');
    const score = Math.max(...candidates.map(candidate => agreement(actual, candidate)));
    return { accepted: score >= threshold, level, score };
  }
  if (level === 'span') {
    const threshold = 'threshold' in spec ? spec.threshold ?? 0.5 : 0.5;
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('span threshold must be between 0 and 1');
    const qa = 'normalization' in spec && spec.normalization === 'qa';
    const score = Math.max(...candidates.map(candidate => spanF1(actual, candidate, qa)));
    if (qa && score < threshold) {
      if (candidates.some(candidate => polarity(actual) !== undefined && polarity(actual) === polarity(candidate)))
        return { accepted: true, level, score, comparison_version: 'qa-polarity/1' };
      if (candidates.some(candidate => containsQaSpan(actual, candidate)))
        return { accepted: true, level, score, comparison_version: 'qa-span-containment/1' };
      const core = Math.max(0, ...candidates.flatMap(candidate =>
        qaAnswerCores(actual, candidate, context.question).map(text => spanF1(text, candidate, true))));
      if (core >= threshold) return { accepted: true, level, score: core, comparison_version: 'qa-span-core/1' };
    }
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
