/**
 * Exact value snapshots for call records, and the typed input features rule induction works on (§3.2, §5.2).
 * Platform-neutral: the store decides where the bytes go.
 */
import { hexDigest } from '../native/hash.js';
import type { ValueRef } from './types.js';

/** A snapshot ready to store: the reference, and the JSON text when it is complete. */
export type Snapshot = { ref: ValueRef; text?: string };

const typeLabel = (value: unknown): string => {try{return value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;}catch{return typeof value;}};

/** Read only data properties. Observation must never execute application getters. */
export function dataEntries(value: object): [string, unknown][] {
  return Object.entries(Object.getOwnPropertyDescriptors(value))
    .filter(([, descriptor]) => descriptor.enumerable && Object.hasOwn(descriptor, 'value'))
    .map(([key, descriptor]) => [key, descriptor.value]);
}

const objectConstructorSource = Function.prototype.toString.call(Object);

function plainPrototype(prototype: object | null): boolean {
  if (prototype === null || prototype === Object.prototype) return true;
  // Workerd, browser, and Node compartments can supply their own Object.prototype.
  if (Object.getPrototypeOf(prototype) !== null) return false;
  const constructor = Object.getOwnPropertyDescriptor(prototype, 'constructor')?.value;
  return typeof constructor === 'function' &&
    Object.getOwnPropertyDescriptor(constructor, 'prototype')?.value === prototype &&
    Function.prototype.toString.call(constructor) === objectConstructorSource;
}

/** Realm-neutral plain-data classification without Symbol.toStringTag or constructor access. */
export function isPlainDataObject(value:unknown):value is Record<string,unknown>{
  return value!==null&&typeof value==='object'&&!Array.isArray(value)&&plainPrototype(Object.getPrototypeOf(value));
}

/** Encode data descriptors directly: JSON.stringify on an application object can execute toJSON. */
function portableText(value: unknown, seen: Set<object>, depth: number): string | undefined {
  if (depth > 64) return undefined;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : undefined;
  if (typeof value !== 'object' || seen.has(value)) return undefined;
  seen.add(value);
  try {
    const descriptors=Object.getOwnPropertyDescriptors(value);
    if (Array.isArray(value)) {
      const length=descriptors.length?.value;
      if(typeof length!=='number'||Object.values(descriptors).filter(item=>item.enumerable).length!==length)return undefined;
      const items:string[]=[];
      for(let index=0;index<length;index++){
        const item=descriptors[String(index)];
        if(!item||!Object.hasOwn(item,'value'))return undefined;
        const text=portableText(item.value,seen,depth+1);
        if(text===undefined)return undefined;
        items.push(text);
      }
      return `[${items.join(',')}]`;
    }
    if(!plainPrototype(Object.getPrototypeOf(value)))return undefined;
    const items:string[]=[];
    for(const [key,item] of Object.entries(descriptors)){
      if(!item.enumerable)continue;
      if(!Object.hasOwn(item,'value'))return undefined;
      const text=portableText(item.value,seen,depth+1);
      if(text===undefined)return undefined;
      items.push(`${JSON.stringify(key)}:${text}`);
    }
    return `{${items.join(',')}}`;
  } finally {seen.delete(value);}
}

const preview = (text: string) => text.length > 200 ? `${text.slice(0, 200)}…` : text;

/** Snapshot `value` for a record: exact JSON within `maxBytes`, else what can be said of it. */
export function snapshot(value: unknown, maxBytes: number, excluded = false): Snapshot {
  if (excluded) return { ref: { complete: false, reason: 'excluded', type: typeLabel(value) } };
  if (value === undefined) value = null;
  let text:string|undefined;
  try {text=portableText(value,new Set(),0);}catch{/* Reflection can fail for revoked proxies. */}
  if(text===undefined)return {ref:{complete:false,reason:'nonportable',type:typeLabel(value)}};
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
      for (const [index,item] of dataEntries(value)) if (/^(0|[1-9]\d*)$/.test(index) && Number(index)<3) visit(item, `${path}[${index}]`,depth+1);
      return;
    }
    if (typeof value === 'object') {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) { put(`type:${path}`, typeLabel(value)); return; }
      put(`keys:${path}`, Object.keys(value).sort().join(','));
      for (const [key, item] of dataEntries(value))
        visit(item, /^[A-Za-z_$][\w$]*$/.test(key) ? `${path}.${key}` : `${path}[${JSON.stringify(key)}]`, depth + 1);
    }
  };
  for (const [name, value] of dataEntries(inputs)) visit(value, name, 0);
  return out;
}
