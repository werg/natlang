/**
 * Built-in natlang programs: the `.nl` files in src/builtin that ship with the runtime (judges, policies, prompts). They
 * load the way a callable folder does (`loadCallableFolder`: the same parsing, type checks and policy), from the sources
 * embedded in the build (builtin/sources.generated.ts), so they work in Node, the npm package and the browser bundle.
 *
 * Runtime code asks for a built-in by name: `builtin('progressJudge')` is the callable; `builtinDefinition(name)` is the
 * kernel definition the runtime invokes with `invokeDefinition` for its own calls, in the exact shape the function had
 * when it was a string literal in TypeScript (same id and name, so traces and call-store keys continue).
 */
import { BUILTIN_SOURCES, builtinNames } from '../builtin/index.js';
import { loadCallableFolder, type ItemRecord, type NatlangRecord, type SourceFiles } from './loader.js';
import { namedCallable, type NatlangCallable } from './callable.js';
import type { CallableDefinition } from './kernel.js';
import { definitionKey } from '../calls/recorder.js';

const ROOT = '/natlang-builtin';

/** The embedded files as loader file access: one flat folder. */
const files: SourceFiles = {
  join: (...parts) => parts.join('/').replace(/\/+/g, '/'),
  dirname: path => path.slice(0, path.lastIndexOf('/')) || '/',
  basename: (path, extension) => { const base = path.slice(path.lastIndexOf('/') + 1);
    return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; },
  extname: path => { const base = path.slice(path.lastIndexOf('/') + 1); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; },
  isFile: path => path.startsWith(`${ROOT}/`) && Object.hasOwn(BUILTIN_SOURCES, path.slice(ROOT.length + 1).replace(/\.nl$/, '')) && path.endsWith('.nl'),
  isDirectory: path => path === ROOT,
  read: path => {
    const source = files.isFile(path) ? BUILTIN_SOURCES[path.slice(ROOT.length + 1, -'.nl'.length)] : undefined;
    if (source === undefined) throw new Error(`no such built-in file ${path}`);
    return source;
  },
  list: path => path === ROOT ? builtinNames().map(name => `${name}.nl`) : [],
  relative: path => path.startsWith(`${ROOT}/`) ? `builtin/${path.slice(ROOT.length + 1)}` : path,
};

let records: Record<string, ItemRecord> | undefined;
/** The built-in programs as records of a callable folder. */
export function builtinRecords(): Record<string, ItemRecord> {
  return records ??= loadCallableFolder(ROOT, files);
}

function recordOf(name: string): NatlangRecord {
  const item = Object.hasOwn(builtinRecords(), name) ? builtinRecords()[name] : undefined;
  if (!item || item.kind !== 'natlang')
    throw new RangeError(`no built-in natlang program named ${JSON.stringify(name)}; the built-ins are ${builtinNames().join(', ')}`);
  return item;
}

const callables = new Map<string, NatlangCallable>();
/** A built-in natlang function by name, callable like any function loaded from a callable folder. */
export function builtin(name: string): NatlangCallable {
  let callable = callables.get(name);
  if (!callable) callables.set(name, callable = namedCallable(name, recordOf(name)));
  return callable;
}

/** The names a built-in had as a string literal before it moved into a `.nl` file; keeping them keeps traces and call-store keys continuous. */
const RUNTIME_IDENTITY: Record<string, { id: string; name: string }> = {
  progressJudge: { id: 'natlang:progress_judge', name: 'progress_judge' },
  compareBehaviors: { id: 'natlang:compareBehaviors', name: 'compareBehaviors' },
};

/** The kernel definition of a built-in, for the runtime's own calls (`invokeDefinition`). */
export function builtinDefinition(name: string): CallableDefinition {
  const record = recordOf(name);
  const identity = RUNTIME_IDENTITY[name] ?? { id: record.id, name: record.name };
  return { id: identity.id, name: identity.name, ...(record.description ? { description: record.description } : {}),
    body: record.instructions.replace(/\n$/, ''),
    params: Object.entries(record.args).map(([raw, type]) => ({ name: raw.replace(/\?$/, ''), type, ...(raw.endsWith('?') ? { optional: true } : {}) })),
    returns: record.returns, ...(record.generic ? { generic: record.generic } : {}), types: record.types, codebase: {}, subtype: record.subtype };
}

/** The revision key a built-in's compilations and calls are stored under (what ran, as run); receipts of a built-in reviewer cite it. */
export function builtinDefinitionKey(name: string): string {
  const definition = builtinDefinition(name);
  return definitionKey({ id: definition.id, body: definition.body, params: definition.params, returns: definition.returns,
    types: definition.types, subtype: definition.subtype });
}
