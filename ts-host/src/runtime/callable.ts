/**
 * Natlang callable objects. One builder turns a record tree into callables for host code and for
 * eval alike: `.nl` items are asynchronous functions invoked through the kernel; TypeScript modules
 * expose their exports (a default function export makes the module itself callable); folders are records.
 * Every natlang function also has the standard `iterateOn` method.
 */
import { RESERVED_CALLABLE_PROPERTIES } from '../compiler/intrinsics.js';
import { APPLY_TO_FOLDER, type FolderHandle } from '../native/scoped-fs.js';
import { currentFrame, runInFrame, type Frame } from './context.js';
import { invokeDefinition, type CallableDefinition, type CaptureCell } from './kernel.js';
import { parseNatlang, PATH_ONLY, type ItemRecord, type ModuleRecord, type NatlangRecord } from './loader.js';
import { moduleInstance } from './modules.js';
import { resolveFrame } from './runtime.js';
import { iterateOn } from './iterate.js';

export const NATLANG_CALLABLE: unique symbol = Symbol.for('natlang.callable') as never;

export type CallableMeta = {
  definition: CallableDefinition;
  kind: 'named' | 'inline';
  /** Frame in which an inline callable was created; a fallback when no frame propagates to the call. */
  created?: Frame;
  /** A frame the callable always runs in (callables handed to an interpreter session's eval). */
  bound?: Frame;
  invoke(args: unknown[], frame: Frame): Promise<unknown>;
};

export type NatlangCallable = ((...args: unknown[]) => Promise<unknown>) & { readonly [NATLANG_CALLABLE]: CallableMeta };

export function isNatlangCallable(value: unknown): value is NatlangCallable {
  return typeof value === 'function' && Object.hasOwn(value, NATLANG_CALLABLE);
}
export function callableMeta(value: unknown): CallableMeta | undefined {
  return isNatlangCallable(value) ? value[NATLANG_CALLABLE] : undefined;
}

/** Define a read-only child attribute, rejecting names that collide with function internals. */
export function defineChild(target: object, name: string, value: unknown): void {
  if (RESERVED_CALLABLE_PROPERTIES.has(name))
    throw new TypeError(`${JSON.stringify(name)} cannot be the name of a natlang child item; rename the source file or export`);
  Object.defineProperty(target, name, { value, enumerable: true, writable: false, configurable: false });
}

export function makeCallable(meta: CallableMeta): NatlangCallable {
  const fn = async function (...args: unknown[]) { return meta.invoke(args, meta.bound ?? resolveFrame(meta.created)); };
  Object.defineProperty(fn, 'name', { value: meta.definition.name });
  Object.defineProperty(fn, NATLANG_CALLABLE, { value: meta });
  Object.defineProperty(fn, 'iterateOn', { value: (initial: unknown, ...fixed: unknown[]) =>
    iterateOn(fn as never, initial, ...fixed).inFrame(meta.bound) });
  return fn as unknown as NatlangCallable;
}

/** The kernel definition of a `.nl` record. */
export function natlangDefinition(record: NatlangRecord): CallableDefinition {
  return { id: record.id, name: record.name, body: record.instructions,
    params: Object.entries(record.args).map(([raw, type]) => ({ name: raw.replace(/\?$/, ''), type, optional: raw.endsWith('?') })),
    returns: record.returns, types: record.types, codebase: record.codebase as Record<string, unknown>,
    subtype: record.subtype, description: record.description, source: record.source, revision: record.revision };
}

/** A callable for a named `.nl` definition, with its callable-folder children as attributes. */
export function namedCallable(name: string, record: NatlangRecord, bound?: Frame): NatlangCallable {
  const definition = natlangDefinition({ ...record, name });
  const fn = makeCallable({ definition, kind: 'named', bound, invoke: (args, frame) => invokeDefinition(frame, definition, args) });
  if (definition.subtype === 'directory-reducer')
    Object.defineProperty(fn, APPLY_TO_FOLDER, { value: async (folder: FolderHandle, args: unknown[]) =>
      invokeDefinition(bound ?? resolveFrame(), definition, [folder, ...args],
        { folder: { transaction: await folder.beginTransaction(true), mode: 'apply' } }) });
  attachChildren(fn, record.codebase, bound);
  return fn;
}

/**
 * Define a natural-language function from `.nl` source text at run time: the dynamic counterpart of a
 * `.nl` file, for apps whose users author functions (notebook cells, wiki cells, generated tools).
 * `codebase` is its callable context, for example the records of a loaded callable folder.
 */
export function defineNatlang(text: string, options: { name?: string; types?: Record<string, string>;
  codebase?: Record<string, ItemRecord> } = {}): NatlangCallable {
  const name = options.name ?? 'fn';
  const record = parseNatlang(`${name}.nl`, text, options.types ?? {}, PATH_ONLY);
  return namedCallable(name, { ...record, id: `nl:defined/${name}@${record.revision}`, source: `defined:${name}`,
    codebase: options.codebase ?? {} });
}

function attachChildren(target: object, codebase: Record<string, ItemRecord>, bound?: Frame): void {
  for (const [name, child] of Object.entries(codebase)) defineChild(target, name, itemValue(child, codebase, bound));
}

/** The value of one item in its folder `level`. */
export function itemValue(item: ItemRecord, level: Record<string, ItemRecord>, bound?: Frame): unknown {
  if (item.kind === 'natlang') return namedCallable(item.name, item, bound);
  if (item.kind === 'namespace') return callableTree(item.codebase, bound);
  return moduleValue(item, level, bound);
}

/** A TypeScript module: callable if it has a default function export; exports and folder children as attributes. */
function moduleValue(record: ModuleRecord, level: Record<string, ItemRecord>, bound?: Frame): unknown {
  const instance = () => moduleInstance(record, level);
  const enter = <T>(fn: () => T): T => bound ? runInFrame(bound, fn) : fn();
  const defaultExport = record.exports.default;
  const node: object = defaultExport?.kind === 'function' ?
    Object.defineProperty((...args: unknown[]) => enter(() => (instance().default as Function)(...args)), 'name', { value: record.name }) :
    Object.create(null);
  if (typeof node === 'function') Object.defineProperty(node, 'iterateOn', { value: (initial: unknown, ...fixed: unknown[]) =>
    iterateOn(node as never, initial, ...fixed) });
  for (const [name, spec] of Object.entries(record.exports)) {
    if (name === 'default') continue;
    if (RESERVED_CALLABLE_PROPERTIES.has(name))
      throw new TypeError(`${JSON.stringify(name)} cannot be the name of a natlang child item; rename the export`);
    Object.defineProperty(node, name, spec.kind === 'function' ?
      { value: (...args: unknown[]) => enter(() => (instance()[name] as Function)(...args)), enumerable: true } :
      { get: () => instance()[name], enumerable: true });
  }
  attachChildren(node, record.codebase, bound);
  return typeof node === 'function' ? node : Object.freeze(node);
}

/** The callable tree for a whole callable folder (for example `natlang.d/`), as a record. */
export function callableTree(codebase: Record<string, ItemRecord>, bound?: Frame): Record<string, unknown> {
  const tree = Object.create(null) as Record<string, unknown>;
  attachChildren(tree, codebase, bound);
  return Object.freeze(tree);
}

/** An inline `nl` instance: one source definition, a fresh instance per evaluation of the tag. */
export function inlineCallable(definition: CallableDefinition, instructions: string,
  captures: Record<string, CaptureCell>, classes?: ReadonlyMap<string, Function>, bound?: Frame): NatlangCallable {
  return makeCallable({ definition, kind: 'inline', created: currentFrame(), bound,
    invoke: (args, frame) => invokeDefinition(frame, definition, args, { captures, instructions, classes,
      manifest: { inline: true } }) });
}
