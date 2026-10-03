/**
 * Contexts (spec/SPEC.md "Contexts"): every natlang function is a curried function of its context, an immutable,
 * content-addressed folder value. A context holds **executable nodes** (`.nl` functions and TypeScript modules, as
 * loader records) and **data entries** (anything else: text, JSON, Neuralese values, soft functions, `.nz` exports).
 *
 * - Executable nodes come from files. The loader registers every record it builds from source files; a context
 *   refuses an executable node that is not registered (`context-new-executable`). An existing node may be replaced by
 *   an edited definition that keeps its signature (`edit`).
 * - Data entries are free.
 * - `fn.in(context)` rebinds a function. The new context is checked against the function's context interface: the
 *   items its instructions name, with their signatures (`context-interface-mismatch`).
 * - A context's ID is a hash of its contents, so a context can never contain itself.
 */
import { hexDigest } from '../native/hash.js';
import { isNeuraleseRef, neuraleseRef } from '../native/neuralese.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { canonicalJson, isSoftFunctionSpec, loadNz, saveNz, NzFileError, type LoadedNz, type NzSaveExport,
  type NzSoftFunctionSpec } from '../native/nz-file.js';
import { formatType, parseType, type Type } from '../native/types.js';
import { inferValueType } from '../native/runtime.js';
import { callableMeta, isNatlangCallable, makeCallable, attachChildren, setRebinder, recordOf, rememberRecord, NATLANG_CALLABLE,
  type NatlangCallable } from './callable.js';
import { invokeDefinition, type CallableDefinition, type CaptureCell } from './kernel.js';
import { parseModule, parseNatlang, PATH_ONLY, loadCallableFolder, registerFileRecords, isFileRecord, nodeKey,
  type ItemRecord, type ModuleRecord, type NatlangRecord, type SourceFiles } from './loader.js';
import { neuraleseSentinel } from '../native/neuralese.js';

export type ContextErrorCode = 'context-interface-mismatch' | 'context-new-executable' | 'context-conflict';

export class ContextError extends Error {
  constructor(readonly code: ContextErrorCode, message: string) { super(`${code}: ${message}`); this.name = 'ContextError'; }
}

// --- File provenance of executable nodes -----------------------------------------------------------------------

export { registerFileRecords, isFileRecord };

const isItemRecord = (value: unknown): value is ItemRecord => !!value && typeof value === 'object' &&
  ['natlang', 'module', 'namespace'].includes((value as { kind?: unknown }).kind as string) && 'codebase' in value;

// --- Soft functions and live captures --------------------------------------------------------------------------

const LIVE: unique symbol = Symbol.for('natlang.live') as never;
/** An explicit capture read at each call and written back after a successful eval (`nl.with({ n: live(n) })`). */
export type LiveCapture = { readonly [LIVE]: true; get(): unknown; set?(value: unknown): void };
/**
 * Mark an explicit capture as live. Compiled code passes the binding's accessors; host code passes a getter (and a
 * setter for write-back).
 */
export function live(get: () => unknown, set?: (value: unknown) => void): LiveCapture {
  if (typeof get !== 'function') throw new TypeError('live(...) takes the accessors of a let binding');
  return Object.freeze({ [LIVE]: true as const, get, ...(set ? { set } : {}) });
}
export const isLiveCapture = (value: unknown): value is LiveCapture => !!value && typeof value === 'object' && LIVE in value;

type SoftMeta = { type: string; body: string; captures: Record<string, unknown>; live: boolean };
const SOFT = new WeakMap<object, SoftMeta>();
/** The stored form of a soft function, when `value` is one. */
export function softFunctionOf(value: unknown): SoftMeta | undefined { return typeof value === 'function' ? SOFT.get(value) : undefined; }

/**
 * A callable Neuralese function: its body is a soft block (shown to the model as a literal), its captures are explicit.
 * Plain capture values are snapshots; `live(...)` captures are read per call and written back. Its calls go only to
 * `context`, its definition site's context.
 */
export function softFunction(spec: { type: string; body: string; captures?: Record<string, unknown>; context?: Context;
  name?: string }): NatlangCallable {
  const type = parseType(spec.type);
  const lambda = type.kind === 'neuralese' ? type.element : type;
  if (lambda.kind !== 'lambda') throw new TypeError(`a soft function needs a function type, got ${spec.type}`);
  const softType = type.kind === 'neuralese' ? spec.type : `Neuralese<${spec.type}>`;
  const context = spec.context ?? Context.empty();
  const captures: Record<string, CaptureCell> = {};
  let hasLive = false;
  for (const [name, value] of Object.entries(spec.captures ?? {})) {
    if (isLiveCapture(value)) {
      hasLive = true;
      const initial = value.get();
      captures[name] = { name, type: inferValueType(initial as never), mutable: !!value.set, get: value.get,
        ...(value.set ? { set: value.set } : {}) };
    } else captures[name] = { name, type: captureType(value), mutable: false, get: () => value };
  }
  Object.assign(captures, dataCells(context, new Set(Object.keys(captures))));
  const definition: CallableDefinition = { id: `nz-fn:${spec.body}`, name: spec.name ?? `soft@${spec.body.slice(4, 16)}`,
    body: neuraleseSentinel(spec.body) + '\n',
    params: lambda.params.fields.map(field => ({ name: field.name, type: formatType(field.type), ...(field.optional ? { optional: true } : {}) })),
    returns: formatType(lambda.returns), types: {}, codebase: context.items as Record<string, unknown>, subtype: 'function',
    contextId: context.id, revision: spec.body.slice(4, 20) };
  const fn = makeCallable({ definition, kind: 'inline', invoke: (args, frame) => invokeDefinition(frame, definition, args,
    { captures, manifest: { inline: true, soft: true } }) });
  attachChildren(fn, context.items);
  SOFT.set(fn, { type: softType, body: spec.body, captures: Object.fromEntries(Object.entries(spec.captures ?? {})
    .map(([name, value]) => [name, isLiveCapture(value) ? value : value])), live: hasLive });
  return fn;
}

function captureType(value: unknown): string {
  if (isNeuraleseRef(value)) return value.$neuralese.type;
  const soft = softFunctionOf(value);
  if (soft) return soft.type;
  if (typeof value === 'function') return 'Live<"function", "function", "">';
  return inferValueType(value as never);
}

// --- Contexts --------------------------------------------------------------------------------------------------

/** Signature of an executable node, compared when rebinding and when editing. */
function signatureOf(record: ItemRecord): string {
  if (record.kind === 'natlang') return canonicalJson({ kind: 'natlang', subtype: record.subtype, args: record.args, returns: record.returns });
  if (record.kind === 'module') return canonicalJson({ kind: 'module', exports: Object.fromEntries(Object.entries(record.exports)
    .map(([name, spec]) => [name, spec.kind === 'function' ? { args: spec.args, returns: spec.returns, async: spec.async } : { type: spec.type ?? null }])) });
  return canonicalJson({ kind: 'namespace', items: Object.fromEntries(Object.entries(record.codebase).map(([name, child]) => [name, signatureOf(child)])) });
}

function recordDigest(record: ItemRecord): unknown {
  const children = Object.fromEntries(Object.entries(record.codebase).sort().map(([name, child]) => [name, recordDigest(child)]));
  return record.kind === 'namespace' ? { namespace: children } : { node: nodeKey(record), children };
}

function dataDigest(value: unknown): unknown {
  if (value instanceof Context) return { context: value.id };
  if (isNeuraleseRef(value)) return { neuralese: value.$neuralese.id, type: value.$neuralese.type };
  const soft = softFunctionOf(value);
  if (soft) return { softFunction: soft.body, type: soft.type, captures: Object.fromEntries(Object.entries(soft.captures)
    .map(([name, item]) => [name, isLiveCapture(item) ? { live: true } : dataDigest(item)])) };
  if (typeof value === 'function') {
    const meta = callableMeta(value);
    return { function: meta ? `${meta.definition.id}@${meta.definition.revision ?? ''}` : String(value) };
  }
  if (value instanceof Uint8Array) return { bytes: hexDigest(value) };
  if (Array.isArray(value)) return value.map(dataDigest);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, dataDigest(item)]));
  return value;
}

/**
 * An immutable, content-addressed context. `items` are executable nodes (loader records, as a callable folder's
 * codebase); `data` maps names or slash-separated paths to data values.
 */
export class Context {
  readonly id: string;
  private constructor(readonly items: Readonly<Record<string, ItemRecord>>, readonly data: Readonly<Record<string, unknown>>) {
    this.id = `ctx1_${hexDigest(canonicalJson({
      items: Object.fromEntries(Object.entries(items).sort().map(([name, record]) => [name, recordDigest(record)])),
      data: Object.fromEntries(Object.entries(data).sort().map(([key, value]) => [key, dataDigest(value)])) })).slice(0, 40)}`;
    Object.freeze(this);
  }

  static empty(): Context { return new Context({}, {}); }

  /** A context over loader records. Every executable node must come from files. */
  static of(items: Record<string, ItemRecord>, data: Record<string, unknown> = {}): Context {
    for (const [name, record] of Object.entries(items))
      if (!isFileRecord(record)) throw new ContextError('context-new-executable',
        `${name} is an executable node that was not loaded from a file; write it to a file in a staged tree first`);
    for (const key of Object.keys(data)) if (Object.hasOwn(items, key))
      throw new ContextError('context-conflict', `${key} is both an executable node and a data entry`);
    return new Context(Object.freeze({ ...items }), Object.freeze({ ...data }));
  }

  /** The context a callable is bound to. */
  static ofCallable(fn: NatlangCallable): Context {
    const meta = callableMeta(fn);
    if (!meta) throw new TypeError('not a natlang callable');
    return Context.of(meta.definition.codebase as Record<string, ItemRecord>, CONTEXT_DATA.get(fn) ?? {});
  }

  /**
   * Load a callable folder as a context: its `.nl` and `.ts` items (registered as file nodes), each `.nz` file's
   * exports as data under the file's name, and other files (Markdown, JSON, text) as data under their relative paths.
   */
  static async fromFolder(dir: string, files: SourceFiles & { readBytes?(path: string): Uint8Array },
    options: { store?: NeuraleseStore } = {}): Promise<Context> {
    const items = loadCallableFolder(dir, files);
    const data: Record<string, unknown> = {};
    const nz = new Map<string, LoadedNz>();
    const loadFile = async (path: string): Promise<LoadedNz> => {
      const found = nz.get(path);
      if (found) return found;
      if (!files.readBytes) throw new Error(`reading ${path} needs binary file access`);
      const loaded = await loadNz(files.readBytes(path), { store: options.store });
      nz.set(path, loaded);
      return loaded;
    };
    const walk = async (folder: string, prefix: string) => {
      for (const entry of files.list(folder).filter(name => !name.startsWith('.')).sort()) {
        const path = files.join(folder, entry), key = prefix ? `${prefix}/${entry}` : entry;
        if (files.isDirectory(path)) { if (entry !== 'node_modules') await walk(path, key); continue; }
        if (entry.endsWith('.nl') || entry === 'types.ts' || (entry.endsWith('.ts') && !entry.endsWith('.d.ts'))) continue;
        if (entry.endsWith('.d.ts') || entry.endsWith('.d.nl.ts') || entry.endsWith('.d.nz.ts')) continue;
        if (entry.endsWith('.nz')) {
          const loaded = await loadFile(path);
          data[key.replace(/\.nz$/, '')] = Object.fromEntries(Object.entries(loaded.exports).map(([name, value]) =>
            [name, isSoftFunctionSpec(value) ? softFunctionFromSpec(value) : value]));
        } else if (entry.endsWith('.json')) data[key] = JSON.parse(files.read(path));
        else data[key] = files.read(path);
      }
    };
    await walk(dir, '');
    return Context.of(items, data);
  }

  /** Executable node and data entry names. */
  get names(): string[] { return [...Object.keys(this.items), ...Object.keys(this.data)].sort(); }

  /**
   * A context with entries added or replaced. A natlang callable or loader record is an executable node and must come
   * from files; everything else (including soft functions, Neuralese values and contexts) is data. A context can be
   * stored as a data entry; the result has a new ID, so a context never contains itself.
   */
  with(entries: Record<string, unknown>): Context {
    const items: Record<string, ItemRecord> = { ...this.items };
    const data: Record<string, unknown> = { ...this.data };
    for (const [key, value] of Object.entries(entries)) {
      const record = isItemRecord(value) ? value : executableRecord(value);
      if (record) {
        if (!isFileRecord(record)) throw new ContextError('context-new-executable',
          `${key} would add an executable node that was not loaded from a file; write it to a file in a staged tree first`);
        if (Object.hasOwn(items, key) && signatureOf(items[key]!) !== signatureOf(record))
          throw new ContextError('context-interface-mismatch', `${key} replaces an executable node with a different signature`);
        if (Object.hasOwn(data, key)) delete data[key];
        items[key] = record;
      } else {
        if (Object.hasOwn(items, key)) throw new ContextError('context-interface-mismatch',
          `${key} is an executable node; replace it with an edited definition (edit), not with data`);
        data[key] = value;
      }
    }
    return new Context(Object.freeze(items), Object.freeze(data));
  }

  /** Data entries removed. Executable nodes are removed with `pick`. */
  without(...keys: string[]): Context {
    const data = { ...this.data };
    for (const key of keys) delete data[key];
    return new Context(this.items, Object.freeze(data));
  }

  /**
   * Replace an executable node with an edited definition. The edit is compiled and must keep the node's signature
   * (`context-interface-mismatch` otherwise). Edits of file nodes count as file nodes.
   */
  edit(name: string, text: string): Context {
    const existing = this.items[name];
    if (!existing || existing.kind === 'namespace') throw new ContextError('context-interface-mismatch', `${name} is not an executable node of this context`);
    const parsed: ItemRecord = existing.kind === 'natlang' ?
      parseNatlang(existing.source.endsWith('.nl') ? existing.source : `${name}.nl`, text, existing.types, PATH_ONLY) :
      parseModule(existing.source.endsWith('.ts') ? existing.source : `${name}.ts`, text, existing.types, PATH_ONLY);
    if (signatureOf(parsed) !== signatureOf(existing))
      throw new ContextError('context-interface-mismatch', `the edit of ${name} changes its signature`);
    const edited = { ...parsed, id: existing.id, programId: existing.programId, codebase: existing.codebase } as ItemRecord;
    registerFileRecords(edited);
    return new Context(Object.freeze({ ...this.items, [name]: edited }), this.data);
  }

  /** The sub-context of the named entries (an entry name, or a data path prefix such as `skills/sql`). */
  pick(...names: string[]): Context {
    const within = (key: string) => names.some(name => key === name || key.startsWith(`${name}/`));
    return new Context(Object.freeze(Object.fromEntries(Object.entries(this.items).filter(([key]) => within(key)))),
      Object.freeze(Object.fromEntries(Object.entries(this.data).filter(([key]) => within(key)))));
  }

  /** The union of two contexts. An entry present in both must be the same entry (`context-conflict`). */
  union(other: Context): Context {
    const items = { ...this.items }, data = { ...this.data };
    for (const [name, record] of Object.entries(other.items)) {
      if (Object.hasOwn(items, name) && canonicalJson(recordDigest(items[name]!)) !== canonicalJson(recordDigest(record)))
        throw new ContextError('context-conflict', `both contexts have a different ${name}`);
      if (Object.hasOwn(data, name)) throw new ContextError('context-conflict', `${name} is data in one context and executable in the other`);
      items[name] = record;
    }
    for (const [key, value] of Object.entries(other.data)) {
      if (Object.hasOwn(items, key)) throw new ContextError('context-conflict', `${key} is data in one context and executable in the other`);
      if (Object.hasOwn(data, key) && canonicalJson(dataDigest(data[key])) !== canonicalJson(dataDigest(value)))
        throw new ContextError('context-conflict', `both contexts have a different ${key}`);
      data[key] = value;
    }
    return new Context(Object.freeze(items), Object.freeze(data));
  }

  /** Contexts reachable through data entries (for diagnostics; content addressing makes self-containment impossible). */
  contains(other: Context): boolean {
    const seen = new Set<string>();
    const visit = (value: unknown): boolean => {
      if (value instanceof Context) {
        if (value.id === other.id) return true;
        if (seen.has(value.id)) return false;
        seen.add(value.id);
        return Object.values(value.data).some(visit);
      }
      return Array.isArray(value) ? value.some(visit) : !!value && typeof value === 'object' && !isNeuraleseRef(value) &&
        Object.values(value).some(visit);
    };
    return Object.values(this.data).some(visit);
  }
}

/** The loader record behind a natlang callable, if it is an executable node (not a soft function or inline value). */
function executableRecord(value: unknown): ItemRecord | undefined {
  if (!isNatlangCallable(value) || softFunctionOf(value)) return;
  const meta = callableMeta(value)!;
  const record = recordOf(value);
  if (record) return record;
  // A named callable without its record, or an inline function written at run time: neither is a file node.
  return { kind: 'natlang', id: meta.definition.id, name: meta.definition.name, source: meta.definition.source ?? '',
    revision: meta.definition.revision ?? `runtime:${meta.definition.id}`, text: '', description: meta.definition.description ?? '',
    args: Object.fromEntries(meta.definition.params.map(param => [`${param.name}${param.optional ? '?' : ''}`, param.type])),
    returns: meta.definition.returns, instructions: meta.definition.body, types: meta.definition.types,
    subtype: meta.definition.subtype, codebase: meta.definition.codebase as Record<string, ItemRecord> };
}


/** Data entries a rebound callable was given, so its context can be recovered. */
const CONTEXT_DATA = new WeakMap<object, Record<string, unknown>>();

/** Read-only scope bindings for a context's data entries: `name` directly, `a/b/c` grouped under `a`. */
function dataCells(context: Context, taken: ReadonlySet<string> = new Set()): Record<string, CaptureCell> {
  const grouped: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(context.data)) {
    const [head, ...rest] = key.split('/');
    if (!head || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(head) || taken.has(head)) continue;
    if (!rest.length) { grouped[head] = value; continue; }
    let node = (grouped[head] && typeof grouped[head] === 'object' && !Array.isArray(grouped[head]) ? grouped[head] : grouped[head] = {}) as Record<string, unknown>;
    for (const part of rest.slice(0, -1)) node = (node[part] && typeof node[part] === 'object' ? node[part] : node[part] = {}) as Record<string, unknown>;
    node[rest.at(-1)!] = value;
  }
  return Object.fromEntries(Object.entries(grouped).map(([name, value]) =>
    [name, { name, type: captureType(value), mutable: false, get: () => value }]));
}

// --- Context interface and rebinding ---------------------------------------------------------------------------

/**
 * The context interface of a definition: the items of its current context that its instructions (or the code of its
 * callable-folder modules) name, with their signatures.
 */
export function contextInterface(definition: CallableDefinition): Record<string, string> {
  const codebase = definition.codebase as Record<string, ItemRecord>;
  const text = definition.body;
  const named = (name: string) => new RegExp(`(?<![A-Za-z0-9_$])${name.replace(/[$]/g, '\\$')}(?![A-Za-z0-9_$])`).test(text);
  return Object.fromEntries(Object.entries(codebase).filter(([name]) => named(name)).map(([name, record]) => [name, signatureOf(record)]));
}

/** `fn.in(context)`: the same definition bound to another context, checked against its context interface. */
export function rebind(fn: NatlangCallable, context: Context): NatlangCallable {
  if (!(context instanceof Context)) throw new TypeError('in(...) takes a Context');
  const meta = callableMeta(fn);
  if (!meta) throw new TypeError('in(...) rebinds a natlang callable');
  const required = contextInterface(meta.definition);
  for (const [name, signature] of Object.entries(required)) {
    const provided = context.items[name];
    if (!provided) throw new ContextError('context-interface-mismatch', `${meta.definition.name} uses ${name}, which the new context lacks`);
    if (signatureOf(provided) !== signature)
      throw new ContextError('context-interface-mismatch', `${meta.definition.name} uses ${name}, whose signature differs in the new context`);
  }
  const definition: CallableDefinition = { ...meta.definition, codebase: context.items as Record<string, unknown>, contextId: context.id };
  const soft = softFunctionOf(fn);
  const ownCaptures = meta.captures ?? {};
  const captures = { ...ownCaptures, ...dataCells(context, new Set(Object.keys(ownCaptures))) };
  const render = meta.instructions;
  const rebound = makeCallable({ ...meta, definition, captures, invoke: (args, frame) => invokeDefinition(frame, definition, args,
    { ...(meta.options ?? {}), captures, ...(render !== undefined ? { instructions: typeof render === 'function' ? render(frame) : render } : {}) }) });
  attachChildren(rebound, context.items, meta.bound);
  if (soft) SOFT.set(rebound, soft);
  CONTEXT_DATA.set(rebound, context.data as Record<string, unknown>);
  const record = recordOf(fn);
  if (record) rememberRecord(rebound, record);
  return rebound;
}
setRebinder((fn, context) => rebind(fn, context as Context));

// --- .nz values ------------------------------------------------------------------------------------------------

/** A callable for a soft-function export. */
export function softFunctionFromSpec(spec: NzSoftFunctionSpec, context?: Context): NatlangCallable {
  return softFunction({ type: spec.type, body: spec.body, captures: { ...spec.captures }, context });
}

/** The exports of a loaded `.nz` file as runtime values (soft functions become callables). */
export function nzExports(loaded: LoadedNz, context?: Context): Record<string, unknown> {
  return Object.fromEntries(Object.entries(loaded.exports).map(([name, value]) =>
    [name, isSoftFunctionSpec(value) ? softFunctionFromSpec(value, context) : value]));
}

/**
 * `save`: write values to `.nz` bytes. Exports are `{ type, value }`, or bare values whose type is inferred (Neuralese
 * values and soft functions carry their types). A soft function with live captures cannot be saved.
 */
export async function save(exports: Record<string, unknown>, options: { store: NeuraleseStore; dialect: string; types?: string;
  skill?: import('../native/nz-file.js').NzSkill; provenance?: Record<string, unknown> }): Promise<Uint8Array> {
  const toSpec = (value: unknown, where: string): unknown => {
    const soft = softFunctionOf(value);
    if (soft) {
      if (soft.live) throw new NzFileError('neuralese-live-capture-save', `${where} has live captures; only snapshot captures can be saved`);
      return { kind: 'soft-function', type: soft.type, body: soft.body,
        captures: Object.fromEntries(Object.entries(soft.captures).map(([name, item]) => [name, toSpec(item, `${where} capture ${name}`)])) };
    }
    if (isLiveCapture(value)) throw new NzFileError('neuralese-live-capture-save', `${where} is a live capture`);
    if (typeof value === 'function') throw new NzFileError('neuralese-file-type', `${where} is a function that is not a soft function`);
    if (isNeuraleseRef(value) || !value || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map((item, index) => toSpec(item, `${where}[${index}]`));
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, toSpec(item, `${where}.${key}`)]));
  };
  const entries: Record<string, NzSaveExport> = {};
  for (const [name, raw] of Object.entries(exports)) {
    const typed = raw && typeof raw === 'object' && !isNeuraleseRef(raw) && 'type' in raw && 'value' in raw &&
      Object.keys(raw).every(key => ['type', 'value', 'description'].includes(key)) ? raw as NzSaveExport : undefined;
    const value = typed ? typed.value : raw;
    const type = typed?.type ?? captureType(value);
    entries[name] = { type, value: toSpec(value, `export ${name}`), ...(typed?.description ? { description: typed.description } : {}) };
  }
  return saveNz(entries, options);
}

export { neuraleseRef, NATLANG_CALLABLE };
export type { Type };
