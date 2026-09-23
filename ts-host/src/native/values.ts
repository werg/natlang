import { checkHost, fitsType, formatType, parseType, TypeEnv } from './types.js';
import type { Type } from './types.js';
import { FileHandle, Folder, FolderHandle, type FolderTransaction } from './scoped-fs.js';

export const MISSING = Symbol('natlang-missing');
export type Missing = typeof MISSING;
/** A live host object (Date, Map, class instance, function, ...) carried by identity. */
export interface LiveObject { readonly __natlangLive?: never }
export type Value = null | boolean | number | string | Value[] | { [key: string]: Value } |
  Pending | Missing | Folder | FolderHandle | FileHandle | LiveObject;
export type Status = 'unreduced' | 'running' | 'quiesced' | 'waiting' | 'done';

/** Live captured binding of an inline lambda: read at each eval, written back when an eval succeeds. */
export type CaptureCell = { name: string; type: string; mutable: boolean; get(): unknown; set?(value: unknown): void };

/** One natural-language function invocation: its typed scope, instructions, and progress. */
export type LambdaNode = { nodeKind: 'lambda'; type: Type; types: Record<string, Type>; typesSrc: Record<string, string>;
  status: Status; note: string; attempts: number; steps: number;
  body: string; originalBody?: string; args: Record<string, Value>; return: Value;
  continuationNote: string; let: Record<string, Value>; letTypes: Record<string, Type>;
  /** Callable context: the record tree of the function's callable folder (see runtime/loader.ts). */
  codebase: Record<string, unknown>; functionName: string;
  subtype: 'function' | 'directory-reducer'; projectTransaction?: FolderTransaction;
  reducerMode: '' | 'apply' | 'direct';
  captures?: Record<string, CaptureCell>;
  /** Constructors for class-typed host contracts. */
  hostClasses?: ReadonlyMap<string, Function> };
export type Pending = LambdaNode;
export type Diagnostic = { path: string; code: string; expected?: string; got?: string };

export class Reject extends Error {
  constructor(readonly diagnostics: Diagnostic[]) {
    super(diagnostics.map(d => [ `${d.path}: ${d.code}`, d.expected ? `expected ${d.expected}` : '',
      d.got ? `got ${d.got}` : '' ].filter(Boolean).join(', ')).join('; '));
  }
}
const reject = (path: string, code: string, expected?: string, got?: string): never => {
  throw new Reject([{ path, code, expected, got }]);
};
const plain = (value: unknown): value is Record<string, unknown> => value !== null &&
  typeof value === 'object' && !Array.isArray(value) &&
  Object.prototype.toString.call(value) === '[object Object]' &&
  (Object.getPrototypeOf(value) === null || Object.getPrototypeOf(Object.getPrototypeOf(value)) === null);
export const isPending = (value: unknown): value is Pending => plain(value) && value.nodeKind === 'lambda';

/** A value that is carried by identity: anything other than portable data, pending nodes, and folder/tree handles. */
export const isLive = (value: unknown): value is LiveObject => typeof value === 'function' ||
  (value !== null && typeof value === 'object' && !Array.isArray(value) && !plain(value) &&
    !(value instanceof Folder) && !(value instanceof FolderHandle) && !(value instanceof FileHandle));

const liveIds = new WeakMap<object, number>();
let nextLiveId = 1;
/** Stable per-process identity for a live value, used in traces and previews. */
export function liveId(value: object): number {
  let id = liveIds.get(value);
  if (id === undefined) { id = nextLiveId++; liveIds.set(value, id); }
  return id;
}
export function liveLabel(value: object): string {
  if (typeof value === 'function') return `function ${(value as Function).name || 'anonymous'}`;
  const tag = Object.prototype.toString.call(value).slice(8, -1);
  const constructor = (value as { constructor?: { name?: string } }).constructor?.name;
  return tag !== 'Object' ? tag : constructor || 'object';
}


function wrapper(raw: unknown): string | undefined {
  if (!plain(raw)) return;
  const keys = Object.keys(raw).filter(k => k === '$lambda');
  return keys.length ? Object.keys(raw).length === 1 ? keys[0] : 'invalid' : undefined;
}

function preview(value: unknown): string {
  if (isPending(value)) return formatType(value.type);
  if (value === MISSING) return 'missing';
  const text = JSON.stringify(value);
  return (text ?? String(value)).slice(0, 45);
}

export function coerce(raw: unknown, type: Type, env: TypeEnv, path = 'value'): Value {
  const wanted = env.resolve(type);
  if (wanted.kind === 'host') {
    if (checkHost(raw, wanted.contract, env.classes)) return raw as Value;
    return reject(path, 'type-mismatch', wanted.name, isLive(raw) ? liveLabel(raw as object) : preview(raw));
  }
  // A function-typed slot holds a live function (for example a natlang callable), never a pending node.
  if (wanted.kind === 'lambda') {
    if (typeof raw === 'function') return raw as Value;
    return reject(path, 'type-mismatch', formatType(type), preview(raw));
  }
  if (wanted.kind === 'union') {
    for (const member of wanted.members) {
      try { return coerce(raw, member, env, path); } catch (error) { if (!(error instanceof Reject)) throw error; }
    }
    return reject(path, 'type-mismatch', formatType(type), preview(raw));
  }
  if (wanted.kind === 'prim') {
    // unknown holds any value, as in TypeScript; code narrows it before relying on a shape.
    if (wanted.name === 'unknown' && raw !== undefined) return raw as Value;
    if ((wanted.name === 'string' || wanted.name === 'Blob') && typeof raw === 'string') return raw;
    if (wanted.name === 'number' && typeof raw === 'number' && Number.isFinite(raw) &&
        (!Number.isInteger(raw) || Number.isSafeInteger(raw))) return raw;
    if (wanted.name === 'boolean' && typeof raw === 'boolean') return raw;
    if (wanted.name === 'null' && raw === null) return null;
    if (wanted.name === 'Folder' && (raw instanceof Folder || raw instanceof FolderHandle)) return raw;
    if (wanted.name === 'FileHandle' && raw instanceof FileHandle) return raw;
    return reject(path, 'type-mismatch', wanted.name, preview(raw));
  }
  if (wanted.kind === 'lit') {
    if (raw === wanted.value) return raw as Value;
    return reject(path, 'type-mismatch', formatType(wanted), preview(raw));
  }
  if (wanted.kind === 'list') {
    if (!Array.isArray(raw)) return reject(path, 'type-mismatch', formatType(type), preview(raw));
    // Array.from builds the array in this realm even when eval produced it in its sandbox realm.
    return Array.from(raw as unknown[], (item, i) => coerce(item, wanted.element, env, `${path}/${i}`));
  }
  if (wanted.kind === 'dict' || wanted.kind === 'record') {
    if (!plain(raw)) return reject(path, 'type-mismatch', formatType(type), preview(raw));
    const out: Record<string, Value> = {};
    for (const [key, value] of Object.entries(raw)) {
      if (key.startsWith('$')) return reject(`${path}/${key}`, 'reserved-key');
      if (wanted.kind === 'dict') out[key] = coerce(value, wanted.element, env, `${path}/${key}`);
      else {
        const field = wanted.fields.find(f => f.name === key);
        if (!field) return reject(`${path}/${key}`, 'unknown-field', formatType(wanted));
        if (value === null && field.optional && !fitsType(parseType('null'), field.type, env)) continue;
        out[key] = coerce(value, field.type, env, `${path}/${key}`);
      }
    }
    return wanted.kind === 'record' ? Object.fromEntries(wanted.fields.filter(f => f.name in out).map(f => [f.name, out[f.name]!])) : out;
  }
  return reject(path, 'type-mismatch', formatType(type), preview(raw));
}


export function buildPending(raw: unknown, env = new TypeEnv(), path = ''): Pending {
  const key = wrapper(raw);
  if (!key || key === 'invalid') return reject(path, 'type-mismatch', 'a single pending wrapper');
  const body = (raw as Record<string, unknown>)[key];
  if (!plain(body)) return reject(path, 'type-mismatch', `a mapping for ${key}`);
  const typesSrc = structuredClone((body.types ?? {}) as Record<string, string>);
  const types = Object.fromEntries(Object.entries(typesSrc).map(([name, text]) => [name, parseType(text)]));
  const inner = env.child(types);
  if (typeof body.type !== 'string') return reject(path, 'type-mismatch', `${key} with a type`);
  const type = parseType(body.type);
  inner.checkNames(type);
  for (const localType of Object.values(types)) inner.checkNames(localType);
  const expected = key.slice(1);
  if (type.kind !== expected) return reject(path, 'type-mismatch', `a ${expected} type`, formatType(type));
  const lambdaKeys = new Set(['type', 'types', 'instructions', 'args', 'return', 'status', 'note',
    'continuation_note', 'codebase', 'let', 'let_types', 'function', 'marks', 'subtype']);
  const extra = Object.keys(body).filter(name => !lambdaKeys.has(name)).sort()[0];
  if (extra) return reject(`${path}/${extra}`, 'unknown-field', 'a Lambda part');
  if (type.kind === 'lambda') {
    const text = body.instructions;
    if (typeof text !== 'string') return reject(path, 'type-mismatch', 'instructions text');
    if (body.args && !plain(body.args) && !(Array.isArray(body.args) && body.args.length === 0))
      return reject(`${path}/args`, 'type-mismatch', formatType(type.params));
    const subtype = String(body.subtype ?? 'function');
    if (!['function', 'directory-reducer'].includes(subtype))
      return reject(`${path}/subtype`, 'type-mismatch', 'function or directory-reducer', subtype);
    const node: LambdaNode = { nodeKind: 'lambda', type, types, typesSrc, status: (body.status ?? 'unreduced') as Status,
      note: String(body.note ?? ''), attempts: 0, steps: 0,
      body: text && !text.endsWith('\n') ? text + '\n' : text, args: {}, return: MISSING,
      continuationNote: String(body.continuation_note ?? ''), let: {}, letTypes: {},
      codebase: plain(body.codebase) ? body.codebase as Record<string, unknown> : {},
      functionName: String(body.function ?? ''),
      subtype: subtype as LambdaNode['subtype'], reducerMode: '' };
    for (const [name, value] of Object.entries((body.args ?? {}) as Record<string, unknown>)) {
      const field = type.params.fields.find(f => f.name === name);
      if (!field) return reject(`${path}/args/${name}`, 'unknown-field');
      node.args[name] = coerce(value, field.type, inner, `${path}/args/${name}`);
    }
    if (Object.hasOwn(body, 'return')) node.return = coerce(body.return, type.returns, inner, `${path}/return`);
    for (const [name, text] of Object.entries((body.let_types ?? {}) as Record<string, string>)) {
      node.letTypes[name] = parseType(text);
      if (Object.hasOwn((body.let ?? {}) as object, name))
        node.let[name] = coerce(((body.let ?? {}) as Record<string, unknown>)[name], node.letTypes[name]!, inner, `${path}/let/${name}`);
    }
    return node;
  }
  return reject(path, 'type-mismatch', `a ${expected} type`);
}

export function loadProgram(raw: unknown): Pending { return buildPending(raw); }

export function problems(value: Value, type: Type, env: TypeEnv, path: string): { holes: Diagnostic[] } {
  const holes: Diagnostic[] = [];
  function walk(item: Value, current: Type, at: string): void {
    if (item === MISSING) { holes.push({ path: at, code: 'hole', expected: formatType(current) }); return; }
    const resolved = env.resolve(current);
    if (resolved.kind === 'union') {
      for (const member of resolved.members) {
        try { coerce(item, member, env, at); walk(item, member, at); return; }
        catch (error) { if (!(error instanceof Reject)) throw error; }
      }
    } else if (resolved.kind === 'host') {
      return;
    } else if (resolved.kind === 'record' && plain(item)) {
      for (const field of resolved.fields) {
        if (!(field.name in item)) { if (!field.optional) holes.push({ path: `${at}/${field.name}`, code: 'hole', expected: field.name }); }
        else walk((item as Record<string, Value>)[field.name] as Value, field.type, `${at}/${field.name}`);
      }
    } else if (resolved.kind === 'list' && Array.isArray(item)) item.forEach((child, i) => walk(child, resolved.element, `${at}/${i}`));
    else if (resolved.kind === 'dict' && plain(item)) for (const [key, child] of Object.entries(item)) walk(child as Value, resolved.element, `${at}/${key}`);
  }
  walk(value, type, path);
  return { holes };
}

export function unboundParts(node: Pending, env: TypeEnv, path: string): Diagnostic[] {
  const inner = env.child(node.types), out: Diagnostic[] = [];
  if (node.nodeKind === 'lambda' && node.type.kind === 'lambda') {
    for (const field of node.type.params.fields) {
      const at = `${path}/args/${field.name}`;
      if (!(field.name in node.args)) { if (!field.optional) out.push({ path: at, code: 'unbound-param', expected: formatType(field.type) }); }
      else out.push(...problems(node.args[field.name]!, field.type, inner, at).holes.map(d => ({ ...d, code: 'unbound-param' })));
    }
  }
  return out;
}

export function dump(value: Value, full = false): unknown {
  if (value === MISSING) return null;
  if (value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle)
    return { $host: { kind: value instanceof FileHandle ? 'file' : 'folder',
      path: value instanceof Folder ? '' : value.relativePath, reconstructable: false } };
  if (isLive(value)) return { $live: { type: liveLabel(value as object), id: liveId(value as object) } };
  if (Array.isArray(value)) return value.map(item => dump(item, full));
  if (value && typeof value === 'object' && !isPending(value))
    return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, dump(child, full)]));
  if (!isPending(value)) return value;
  const body: Record<string, unknown> = { type: formatType(value.type) };
  if (Object.keys(value.typesSrc).length) body.types = value.typesSrc;
  if (value.status !== 'unreduced') body.status = value.status;
  if (value.note) body.note = value.note;
  {
    body.instructions = value.body;
    if (Object.keys(value.args).length) body.args = dump(value.args, full);
    if (value.return !== MISSING) body.return = dump(value.return, full);
    if (value.continuationNote) body.continuation_note = value.continuationNote;
    if (Object.keys(value.let).length) body.let = dump(value.let, full);
    if (full && Object.keys(value.letTypes).length)
      body.let_types = Object.fromEntries(Object.entries(value.letTypes).map(([k, t]) => [k, formatType(t)]));
    if (full && Object.keys(value.codebase).length) body.codebase = value.codebase;
    if (value.functionName) body.function = value.functionName;
    if (value.subtype !== 'function') body.subtype = value.subtype;
  }
  return { $lambda: body };
}

export function dumpState(value: Value): unknown { return dump(value, true); }
