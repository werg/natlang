/**
 * Natlang callable objects. One builder turns a record tree into callables for host code and for
 * eval alike: `.nl` items are asynchronous functions invoked through the kernel; TypeScript modules
 * expose their exports (a default function export makes the module itself callable); folders are records.
 * Every natlang function also has the standard `iterateOn` method.
 */
import { RESERVED_CALLABLE_PROPERTIES } from '../compiler/intrinsics.js';
import { APPLY_TO_FOLDER, type FolderHandle } from '../native/scoped-fs.js';
import { currentFrame, runInFrame, startedCalls, type Frame } from './context.js';
import { invokeDefinition, type CallableDefinition, type CaptureCell } from './kernel.js';
import { parseNatlang, PATH_ONLY, type ItemRecord, type ModuleRecord, type NatlangRecord } from './loader.js';
import { moduleInstance } from './modules.js';
import { resolveFrame } from './runtime.js';
import { iterateOn } from './iterate.js';
import { traceFor } from '../native/graph.js';

export const NATLANG_CALLABLE: unique symbol = Symbol.for('natlang.callable') as never;

export type CallableMeta = {
  definition: CallableDefinition;
  kind: 'named' | 'inline';
  /** Frame in which an inline callable was created; a fallback when no frame propagates to the call. */
  created?: Frame;
  /** A frame the callable always runs in (callables handed to an interpreter session's eval). */
  bound?: Frame;
  /** Capture cells an inline or soft function was created with, kept so rebinding preserves them. */
  captures?: Record<string, CaptureCell>;
  /** Invocation options other than captures (manifest, classes), kept for rebinding. */
  options?: import('./kernel.js').InvokeOptions;
  /** Instructions of an inline function, rendered per call; kept for rebinding. */
  instructions?: string | ((frame: Frame) => string);
  /** A compiler-owned inline `nl` recipe. Only these generated callables support `.with(record)`. */
  rebindInline?: (captures: Record<string, unknown>, origin?: import('./lowered.js').InlineInstructionOrigin,
    sources?: Record<string, 'input' | 'local' | 'block' | 'handle'>) => NatlangCallable;
  invoke(args: unknown[], frame: Frame): Promise<unknown>;
};

/**
 * Installed by runtime/contexts.ts: `fn.in(context)`. Kept on a global symbol, not a module binding: contexts.ts and
 * this module import each other through the kernel, and either may finish loading first (no module bindings, no TDZ).
 */
type Rebinder = (fn: NatlangCallable, context: unknown) => NatlangCallable;
export function setRebinder(next: Rebinder): void { (globalThis as Record<symbol, unknown>)[Symbol.for('natlang.rebinder')] = next; }

/**
 * Installed by runtime/contexts.ts: scope cells for a record's default-bound data entries (`.nz` files of its companion
 * folder), with soft-function exports as callables. Same global-symbol arrangement as the rebinder.
 */
type DataCells = (data: Record<string, unknown>) => { captures: Record<string, CaptureCell>; skillFiles?: Record<string, string | Uint8Array> };
export function setDataCells(next: DataCells): void { (globalThis as Record<symbol, unknown>)[Symbol.for('natlang.data-cells')] = next; }
function dataCellsOf(record: NatlangRecord): ReturnType<DataCells> | undefined {
  if (!record.contextData) return;
  const make = (globalThis as Record<symbol, unknown>)[Symbol.for('natlang.data-cells')] as DataCells | undefined;
  if (!make) throw new Error('internal error: context data needs runtime/contexts.js to be loaded');
  return make(record.contextData);
}

/** Loader records of named callables, so a context can take a callable as an executable node. */
function records(): WeakMap<object, ItemRecord> {
  // A hoisted function, not a module constant: see REBINDER on import order.
  return (records as unknown as { map?: WeakMap<object, ItemRecord> }).map ??= new WeakMap();
}
export function recordOf(fn: unknown): ItemRecord | undefined { return typeof fn === 'function' ? records().get(fn) : undefined; }
export function rememberRecord(fn: object, record: ItemRecord): void { records().set(fn, record); }

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
  // A call model code starts and never awaits must not end the host as an unhandled rejection (see invokeDefinition).
  const fn = function (...args: unknown[]) {
    let started: unknown;
    const call = (async () => {
      const bound = meta.bound ?? resolveFrame(meta.created);
      // Called from an eval of the invocation it is bound to: the call stops with that eval (spec: Eval), and reports
      // its decision readout to a decide(...) around it.
      const current = currentFrame();
      const same = current && current.task === bound.task && current.parentCallId === bound.parentCallId;
      const signal = same && current.signal && current.signal !== bound.signal ? current.signal : undefined;
      const readout = same && current.readout !== bound.readout ? current.readout : undefined;
      const frame = signal || readout ? { ...bound, ...(signal ? { signal } : {}), ...(readout ? { readout } : {}) } : bound;
      return started = meta.invoke(args, frame);
    })();
    if (started && typeof started === 'object') startedCalls.set(call, started);
    call.catch(() => {});
    return call;
  };
  Object.defineProperty(fn, 'name', { value: meta.definition.name });
  Object.defineProperty(fn, NATLANG_CALLABLE, { value: meta });
  Object.defineProperty(fn, 'iterateOn', { value: (initial: unknown, ...fixed: unknown[]) =>
    iterateOn(fn as never, initial, ...fixed).inFrame(meta.bound) });
  if (meta.rebindInline) Object.defineProperty(fn, 'with', { value: (captures: Record<string, unknown>) =>
    meta.rebindInline!(captures) });
  Object.defineProperty(fn, 'in', { value: (context: unknown) => {
    const rebinder = (globalThis as Record<symbol, unknown>)[Symbol.for('natlang.rebinder')] as Rebinder | undefined;
    if (!rebinder) throw new Error('contexts are not available in this runtime build');
    return rebinder(fn as unknown as NatlangCallable, context);
  } });
  return fn as unknown as NatlangCallable;
}

/** The kernel definition of a `.nl` record. */
export function natlangDefinition(record: NatlangRecord): CallableDefinition {
  return { programId: record.programId, id: record.id, name: record.name, body: record.instructions,
    params: Object.entries(record.args).map(([raw, type]) => ({ name: raw.replace(/\?$/, ''), type, optional: raw.endsWith('?') })),
    returns: record.returns, types: record.types, codebase: record.codebase as Record<string, unknown>,
    subtype: record.subtype, ...(record.readout ? { readout: record.readout } : {}), ...(record.model ? { model: record.model } : {}), description: record.description, source: record.source, revision: record.revision };
}

/** A callable for a named `.nl` definition, with its callable-folder children as attributes. */
export function namedCallable(name: string, record: NatlangRecord, bound?: Frame): NatlangCallable {
  const definition = natlangDefinition({ ...record, name });
  // Data entries of the companion folder (its .nz files) are bound by default, as read-only scope bindings.
  let cells: ReturnType<DataCells> | undefined | null = null;
  const dataBinding = () => cells === null ? (cells = dataCellsOf(record)) : cells;
  const captures = () => dataBinding()?.captures;
  const meta: CallableMeta = { definition, kind: 'named', bound,
    invoke: (args, frame) => invokeDefinition(frame, definition, args, record.contextData ? { captures: captures()!, ...(dataBinding()?.skillFiles ? { skillFiles: dataBinding()!.skillFiles } : {}) } : undefined) };
  if (record.contextData) Object.defineProperty(meta, 'captures', { get: captures, enumerable: true });
  const fn = makeCallable(meta);
  if (definition.subtype === 'directory-reducer')
    Object.defineProperty(fn, APPLY_TO_FOLDER, { value: async (folder: FolderHandle, args: unknown[]) =>
      invokeDefinition(bound ?? resolveFrame(), definition, [folder, ...args],
        { folder: { transaction: await folder.beginTransaction(true), mode: 'apply' } }) });
  attachChildren(fn, record.codebase, bound);
  rememberRecord(fn, record);
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

export function attachChildren(target: object, codebase: Record<string, ItemRecord>, bound?: Frame): void {
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
  const invoke = (exportName: string, args: unknown[]) => enter(() => {
    recordSkillHelperUse(record.source, exportName, bound ?? currentFrame());
    const target = exportName === 'default' ? instance().default : instance()[exportName];
    return (target as Function)(...args);
  });
  const defaultExport = record.exports.default;
  const node: object = defaultExport?.kind === 'function' ?
    Object.defineProperty((...args: unknown[]) => invoke('default', args), 'name', { value: record.name }) :
    Object.create(null);
  if (typeof node === 'function') Object.defineProperty(node, 'iterateOn', { value: (initial: unknown, ...fixed: unknown[]) =>
    iterateOn(node as never, initial, ...fixed) });
  for (const [name, spec] of Object.entries(record.exports)) {
    if (name === 'default') continue;
    if (RESERVED_CALLABLE_PROPERTIES.has(name))
      throw new TypeError(`${JSON.stringify(name)} cannot be the name of a natlang child item; rename the export`);
    Object.defineProperty(node, name, spec.kind === 'function' ?
      { value: (...args: unknown[]) => invoke(name, args), enumerable: true } :
      { get: () => instance()[name], enumerable: true });
  }
  attachChildren(node, record.codebase, bound);
  return typeof node === 'function' ? node : Object.freeze(node);
}

/** Record helper invocation without arguments/results; a source path must be inside a bound skill helper tree. */
function recordSkillHelperUse(source: string, exportName: string, frame?: Frame): void {
  const normalized = source.replace(/\\/g, '/').replace(/^\.\//, '');
  const match = /(?:^|\/)skills\/([a-z0-9]+(?:-[a-z0-9]+)*)\/(helpers\/.+)$/.exec(normalized);
  if (!match) return;
  const trace = traceFor(frame?.parentCallId);
  if (!trace) return;
  const skillName = match[1]!;
  const offered = trace.events.find(event => event.kind === 'skill_use' && event.phase === 'offered' && event.skill_name === skillName);
  if (!offered || typeof offered.skill_revision !== 'string') return;
  trace.emit('skill_use', { phase: 'helper_invoked', skill_name: skillName,
    skill_revision: offered.skill_revision, invocation_id: trace.events[0]?.run_id ?? null,
    path: match[2]!, helper_export: exportName });
}

/** The callable tree for a whole callable folder (for example `natlang.d/`), as a record. */
export function callableTree(codebase: Record<string, ItemRecord>, bound?: Frame): Record<string, unknown> {
  const tree = Object.create(null) as Record<string, unknown>;
  attachChildren(tree, codebase, bound);
  return Object.freeze(tree);
}

/** An inline `nl` instance: one source definition, a fresh instance per evaluation of the tag. */
export function inlineCallable(definition: CallableDefinition, instructions: string | ((frame: Frame) => string),
  captures: Record<string, CaptureCell>, classes?: ReadonlyMap<string, Function>, bound?: Frame, provenance: Record<string, unknown> = {},
  rebindInline?: CallableMeta['rebindInline']): NatlangCallable {
  const manifest = { ...provenance, inline: true };
  return makeCallable({ definition, kind: 'inline', created: currentFrame(), bound, captures, instructions,
    ...(rebindInline ? { rebindInline } : {}),
    options: { classes, manifest },
    invoke: (args, frame) => invokeDefinition(frame, definition, args, { captures, instructions: typeof instructions === 'function' ? instructions(frame) : instructions, classes,
      ...(frame.skillFiles ? { skillFiles: frame.skillFiles } : {}),
      manifest }) });
}
