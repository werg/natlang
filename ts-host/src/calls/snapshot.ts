/**
 * Exact value snapshots for call records, and the typed input features rule induction works on (§3.2, §5.2).
 * Platform-neutral: the store decides where the bytes go.
 */
import { hexDigest } from '../native/hash.js';
import type { ValueRef } from './types.js';

/** A snapshot ready to store: the reference, and the JSON text when it is complete. */
export type Snapshot = { ref: ValueRef; text?: string };

const typeLabel = (value: unknown): string => {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value !== 'object') return typeof value;
  const name = (value as object).constructor?.name;
  return name && name !== 'Object' ? name : 'object';
};

/** Plain JSON data only: no class instances, functions, cycles, non-finite numbers or undefined fields. */
function portable(value: unknown, seen: Set<object>, depth: number): boolean {
  if (depth > 64) return false;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || seen.has(value)) return false;
  seen.add(value);
  try {
    if (Array.isArray(value)) return value.every(item => portable(item, seen, depth + 1));
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return false;
    return Object.entries(value).every(([, item]) => item !== undefined && portable(item, seen, depth + 1));
  } finally { seen.delete(value); }
}

const preview = (text: string) => text.length > 200 ? `${text.slice(0, 200)}…` : text;

/** Snapshot `value` for a record: exact JSON within `maxBytes`, else what can be said of it. */
export function snapshot(value: unknown, maxBytes: number, excluded = false): Snapshot {
  if (excluded) return { ref: { complete: false, reason: 'excluded', type: typeLabel(value) } };
  if (value === undefined) value = null;
  if (!portable(value, new Set(), 0)) {
    let shown: string | undefined;
    try { shown = preview(String(value)); } catch { /* opaque */ }
    return { ref: { complete: false, reason: 'nonportable', type: typeLabel(value), ...(shown ? { preview: shown } : {}) } };
  }
  const text = JSON.stringify(value);
  const bytes = new TextEncoder().encode(text).byteLength;
  const hash = hexDigest(text);
  if (bytes > maxBytes) return { ref: { complete: false, reason: 'oversize', hash, bytes, type: typeLabel(value), preview: preview(text) } };
  return { ref: { complete: true, hash, bytes }, text };
}

// --- Features -----------------------------------------------------------------------------------------------------

const MAX_FEATURES = 600;
const MAX_TOKENS_PER_STRING = 120;
/** A string this short is also a categorical value (an enum, a keyword, a short command). */
const CATEGORICAL_CHARS = 48;
export const VALUE_CLASSES: Record<string, RegExp> = {
  number: /^\s*-?\d+(?:\.\d+)?\s*$/, date: /\b\d{4}-\d{2}-\d{2}\b/, url: /\bhttps?:\/\/\S+/i,
  email: /\b[\w.+-]+@[\w-]+\.[\w.]+\b/, path: /(?:^|\s)(?:\.{0,2}\/)?[\w.-]+\/[\w./-]+/, multiline: /\n/, empty: /^\s*$/,
};

/** Lowercase word tokens of a string, in first-seen order. */
export function tokens(text: string): string[] {
  return [...new Set(text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [])].slice(0, MAX_TOKENS_PER_STRING);
}

/**
 * Typed features of a call's named inputs. Keys name a predicate family and the accessor path the guard would use:
 * `str:request` (short strings), `len:request`, `tok:request:refund`, `cls:request:url`, `num:limit`, `bool:dryRun`,
 * `null:note`, `arr:items` (length), `keys:options` (sorted key list).
 */
export function inputFeatures(inputs: Record<string, unknown>): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  const put = (key: string, value: string | number | boolean) => { if (Object.keys(out).length < MAX_FEATURES) out[key] = value; };
  const visit = (value: unknown, path: string, depth: number) => {
    if (depth > 4 || Object.keys(out).length >= MAX_FEATURES) return;
    if (value === null || value === undefined) { put(`null:${path}`, true); return; }
    if (typeof value === 'string') {
      put(`len:${path}`, value.length);
      if (value.length <= CATEGORICAL_CHARS) put(`str:${path}`, value);
      for (const token of tokens(value)) put(`tok:${path}:${token}`, true);
      for (const [name, pattern] of Object.entries(VALUE_CLASSES)) if (pattern.test(value)) put(`cls:${path}:${name}`, true);
      return;
    }
    if (typeof value === 'number') { put(`num:${path}`, value); return; }
    if (typeof value === 'boolean') { put(`bool:${path}`, value); return; }
    if (Array.isArray(value)) {
      put(`arr:${path}`, value.length);
      value.slice(0, 3).forEach((item, index) => visit(item, `${path}[${index}]`, depth + 1));
      return;
    }
    if (typeof value === 'object') {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) { put(`type:${path}`, typeLabel(value)); return; }
      put(`keys:${path}`, Object.keys(value).sort().join(','));
      for (const [key, item] of Object.entries(value))
        visit(item, /^[A-Za-z_$][\w$]*$/.test(key) ? `${path}.${key}` : `${path}[${JSON.stringify(key)}]`, depth + 1);
    }
  };
  for (const [name, value] of Object.entries(inputs)) visit(value, name, 0);
  return out;
}
