import { inlineDescriptor } from '../adaptation/inventory.js';
import { fingerprint } from '../adaptation/identity.js';
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
  return { programId: plan.programId, id: plan.definitionId, name: `nl@${plan.sourceSpan.file.split('/').at(-1)}:${plan.sourceSpan.line}`,
    body: plan.instructions,
    params: plan.parameters.map(parameter => ({ name: parameter.name, type: targetType(parameter.type) })),
    ...(plan.openParameters ? { openParameters: true } : {}),
    returns: targetType(plan.returns), types, codebase, subtype: 'function',
    revision: plan.inheritedCodebaseRevision || undefined, source: plan.sourceSpan.file };
}

/** Join cooked template strings with interpolated values, as JavaScript would. */
export function interpolate(strings: readonly string[], values: readonly unknown[]): string {
  let text = strings[0] ?? '';
  values.forEach((value, index) => {
    text += interpolationText(value);
    text += strings[index + 1] ?? '';
  });
  return text.endsWith('\n') ? text : `${text}\n`;
}

function interpolationText(value: unknown): string {
  return typeof value === 'string' ? value : value && typeof value === 'object' ?
    (() => { try { return JSON.stringify(value); } catch { return String(value); } })() : String(value);
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
  // Capture interpolation values once when the tag is evaluated, then resolve static text at invocation.
  const renderedValues = values.map(interpolationText);
  const render = (frame: import('./context.js').Frame) => {
    const view = frame.task.programView;
    const effectivePlan = (!plan.programId || plan.programId === view.program?.id) ? view.inlineRevision(plan) : plan;
    const descriptor = view.component(plan.definitionId);
    if (view.binding && descriptor && (!plan.programId || plan.programId === view.program?.id) && plan.adaptation && !view.patched(plan.sourceSpan.file)) {
      const actual = inlineDescriptor(view.program!.id, plan);
      if (actual.baselineHash !== descriptor.baselineHash ||
        fingerprint({ ...actual.contract, servicesHash: descriptor.contract.servicesHash }) !== descriptor.contractHash)
        throw new Error('compiled inline site differs from bound program; rebuild/revalidate');
    }
    const replacement = view.value(plan.definitionId, plan.programId);
    return interpolate(replacement?.kind === 'lambda.instructions' ? replacement.template.segments : effectivePlan.strings, renderedValues);
  };
  return inlineCallable(planDefinition(plan, context), render, captures, undefined, bound);
}

/** A named `.nl` import compiled into a module: the definition record embedded at build time. */
export function named(name: string, record: NatlangRecord): NatlangCallable { return namedCallable(name, record); }

/** A callable folder such as `natlang.d/`, as a record of callables. */
export function folder(codebase: Record<string, ItemRecord>): Record<string, unknown> { return callableTree(codebase); }

/** Runtime half of the finite-iteration policy: `for (x of y)` becomes `for (x of finite(y, "y"))`. */
export function finite<T>(source: Iterable<T>, label?: string): Iterable<T> {
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
  throw new TypeError(notIterable(source, label));
}

/** Why a `for ... of` source was refused, with the usual way to give the loop what it needs. */
function notIterable(source: unknown, label: string | undefined): string {
  const loop = label ? `\`for (… of ${label})\`` : '`for ... of`';
  const lead = `${loop} iterates an array, string, Map or Set`;
  if (source === null || source === undefined)
    return `${lead}, but got ${source}. A value that can be missing, such as String.match's result when nothing ` +
      `matches, needs a fallback: \`for (… of ${label ?? 'value'} ?? [])\`.`;
  if (typeof source === 'object' && typeof (source as { [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function')
    return `${lead}; spread other iterables into an array first: \`for (… of [...${label ?? 'value'}])\`.`;
  if (typeof source === 'object')
    return `${lead}, but got a plain object; iterate Object.keys, Object.values or Object.entries of it.`;
  return `${lead}, but got a ${typeof source}.`;
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
