/**
 * Runtime support targeted by the natlang compiler's lowering. Compiled modules import this as
 * `__natlang`; eval programs receive the same functions through their scope. Nothing here parses or
 * evaluates raw `nl` text: an uncompiled `nl` fails loudly.
 */
import type { InlineLambdaPlan } from '../compiler/inline.js';
import type { TargetDescriptor } from '../compiler/targets.js';
import { NATLANG_COMPILE_VERSION } from '../compiler/intrinsics.js';
import { bindAwait, guard } from './context.js';
import { callableTree, inlineCallable, namedCallable, type NatlangCallable } from './callable.js';
import type { ItemRecord, NatlangRecord } from './loader.js';
import type { CallableDefinition, CaptureCell } from './kernel.js';
import { Iteration } from './iterate.js';
import { resolveFrame } from './runtime.js';

export { bindAwait, guard };

/** Portable natlang type text for a target descriptor; host objects become `Live<...>`. */
export function targetType(target: TargetDescriptor): string {
  if (target.natlang) return target.natlang;
  const host = target.host;
  const kind = host?.kind ?? 'any';
  const detail = host?.kind === 'tag' ? host.tag : host?.kind === 'class' ? host.name :
    host?.kind === 'shape' ? host.members.join(',') : '';
  return `Live<${JSON.stringify(target.text)}, ${JSON.stringify(kind === 'folder' || kind === 'file' ? 'any' : kind)}, ${JSON.stringify(detail)}>`;
}

export function planDefinition(plan: InlineLambdaPlan, codebase: Record<string, unknown> = {}): CallableDefinition {
  const types: Record<string, string> = {};
  for (const target of [plan.returns, ...plan.parameters.map(parameter => parameter.type), ...plan.captures.map(capture => capture.type)])
    Object.assign(types, target.aliases);
  return { id: plan.definitionId, name: `nl@${plan.sourceSpan.file.split('/').at(-1)}:${plan.sourceSpan.line}`,
    body: plan.instructions,
    params: plan.parameters.map(parameter => ({ name: parameter.name, type: targetType(parameter.type) })),
    returns: targetType(plan.returns), types, codebase, subtype: 'function',
    revision: plan.inheritedCodebaseRevision || undefined, source: plan.sourceSpan.file };
}

/** Join cooked template strings with interpolated values, as JavaScript would. */
export function interpolate(strings: readonly string[], values: readonly unknown[]): string {
  let text = strings[0] ?? '';
  values.forEach((value, index) => {
    text += typeof value === 'string' ? value : value && typeof value === 'object' ?
      (() => { try { return JSON.stringify(value); } catch { return String(value); } })() : String(value);
    text += strings[index + 1] ?? '';
  });
  return text.endsWith('\n') ? text : `${text}\n`;
}

export type CaptureAccessors = Record<string, readonly [() => unknown, ((value: unknown) => void)?]>;

/** Create an inline natlang callable instance for a compiled `nl` expression. */
export function inline(plan: InlineLambdaPlan, values: readonly unknown[], accessors: CaptureAccessors,
  context?: Record<string, unknown>, version: number = NATLANG_COMPILE_VERSION, bound?: import('./context.js').Frame): NatlangCallable {
  if (version !== NATLANG_COMPILE_VERSION)
    throw new Error(`this module was compiled for natlang output version ${version}; rebuild it with natlang build`);
  const captures: Record<string, CaptureCell> = {};
  for (const capture of plan.captures) {
    const accessor = accessors[capture.name];
    if (!accessor) continue;
    captures[capture.name] = { name: capture.name, type: targetType(capture.type), mutable: capture.mutable && !!accessor[1],
      get: accessor[0], ...(capture.mutable && accessor[1] ? { set: accessor[1] } : {}) };
  }
  return inlineCallable(planDefinition(plan, context), interpolate(plan.strings, values), captures, undefined, bound);
}

/** A named `.nl` import compiled into a module: the definition record embedded at build time. */
export function named(name: string, record: NatlangRecord): NatlangCallable { return namedCallable(name, record); }

/** A callable folder such as `natlang.d/`, as a record of callables. */
export function folder(codebase: Record<string, ItemRecord>): Record<string, unknown> { return callableTree(codebase); }

/** Runtime half of the finite-iteration policy: `for (x of y)` becomes `for (x of finite(y))`. */
export function finite<T>(source: Iterable<T>): Iterable<T> {
  if (typeof source === 'string') return source;
  if (Array.isArray(source)) {
    const array = source, length = array.length;
    return { [Symbol.iterator]: () => {
      let index = 0;
      return { next: (): IteratorResult<T> => {
        if (array.length > length) throw new RangeError('the array grew while it was being iterated; build a new array instead');
        return index < Math.min(length, array.length) ? { value: array[index++]!, done: false } : { value: undefined, done: true };
      } };
    } };
  }
  const tag = Object.prototype.toString.call(source);
  if (tag === '[object Map]' || tag === '[object Set]') {
    const collection = source as unknown as { size: number; [Symbol.iterator](): Iterator<T> };
    const size = collection.size;
    return { [Symbol.iterator]: () => {
      const inner = collection[Symbol.iterator]();
      return { next: () => {
        if (collection.size > size) throw new RangeError('the collection grew while it was being iterated');
        return inner.next();
      } };
    } };
  }
  throw new TypeError('`for ... of` in natlang callable code must iterate an array, string, Map, or Set');
}

/** Attach the compiler's call-site identity to an `iterateOn` expression. */
export function site<T>(id: string, iteration: T): T {
  if (iteration instanceof Iteration) iteration.withCompilerSite(id);
  return iteration;
}

/** `import { wiki } from 'natlang:services'` resolves each access against the current task. */
export function service(name: string): object {
  const resolve = () => {
    const found = resolveFrame().task.services[name];
    if (!found) throw new Error(`the natlang service ${JSON.stringify(name)} is not provided to this task`);
    return found as Record<PropertyKey, unknown>;
  };
  return new Proxy(Object.create(null), {
    get: (_, property) => { const target = resolve(); const value = target[property];
      return typeof value === 'function' ? (value as Function).bind(target) : value; },
    has: (_, property) => property in resolve(),
    ownKeys: () => Reflect.ownKeys(resolve()),
    getOwnPropertyDescriptor: (_, property) => {
      const descriptor = Reflect.getOwnPropertyDescriptor(resolve(), property);
      return descriptor ? { ...descriptor, configurable: true } : undefined;
    },
  });
}

/** The raw `nl` export: reached only when a module was not compiled by the natlang compiler. */
export function uncompiled(): never {
  throw new Error('This `nl` expression was not compiled by the natlang compiler. Build the project with `natlang build` ' +
    '(or run it with `natlang run`); raw `nl` text is never evaluated at runtime.');
}
