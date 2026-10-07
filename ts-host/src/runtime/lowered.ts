import { inlineDescriptor } from '../adaptation/inventory.js';
import { canonical, fingerprint } from '../adaptation/identity.js';
/**
 * Runtime support targeted by the natlang compiler's lowering. Compiled modules import this as
 * `__natlang`; eval programs receive the same functions through their scope. Nothing here parses or
 * evaluates raw `nl` text: an uncompiled `nl` fails loudly.
 */
import type { InlineLambdaPlan } from '../compiler/inline.js';
import { portablePrimitiveLiteral, type TargetDescriptor } from '../compiler/targets.js';
import { NATLANG_COMPILE_VERSION } from '../compiler/intrinsics.js';
import { bindAwait, guard } from './context.js';
import { callableTree, inlineCallable, namedCallable, type NatlangCallable } from './callable.js';
import { registerFileRecords, type ItemRecord, type NatlangRecord } from './loader.js';
import type { CallableDefinition, CaptureCell } from './kernel.js';
import { Iteration } from './iterate.js';
import { parseType } from '../native/types.js';
import { fromBase64, importedBlocks, loadNzSync, registerImportedBlocks } from '../native/nz-file.js';
import { live, nzExports, softFunction } from './contexts.js';
import { resolveFrame } from './runtime.js';
import { readNeuraleseForCurrentTask } from '../neuralese/combinators.js';

export { bindAwait, guard };

/** Compiler target for automatic JavaScript string coercion of a Neuralese reference. */
export const readNeuralese = readNeuraleseForCurrentTask;

/** Portable natlang type text for a target descriptor; host objects become `Live<...>`. */
export function targetType(target: TargetDescriptor): string {
  if (target.natlang) return target.natlang;
  const host = target.host;
  const kind = host?.kind ?? 'any';
  const detail = host?.kind === 'tag' ? host.tag : host?.kind === 'class' ? host.name :
    host?.kind === 'shape' ? host.members.join(',') : '';
  return `Live<${JSON.stringify(target.text)}, ${JSON.stringify(kind === 'folder' || kind === 'file' ? 'any' : kind)}, ${JSON.stringify(detail)}>`;
}

/** Type aliases used by every part of an inline callable signature or capture. */
function planTypes(plan: InlineLambdaPlan): Record<string, string> {
  const types: Record<string, string> = {};
  for (const target of [plan.returns, ...plan.parameters.map(parameter => parameter.type), ...plan.captures.map(capture => capture.type)])
    Object.assign(types, target.aliases);
  return types;
}

export function planDefinition(plan: InlineLambdaPlan, codebase: Record<string, unknown> = {}): CallableDefinition {
  return { programId: plan.programId, id: plan.definitionId, name: `nl@${plan.sourceSpan.file.split('/').at(-1)}:${plan.sourceSpan.line}`,
    body: plan.instructions,
    params: plan.parameters.map(parameter => ({ name: parameter.name, type: targetType(parameter.type) })),
    ...(plan.openParameters ? { openParameters: true } : {}),
    returns: targetType(plan.returns), types: planTypes(plan), codebase, subtype: 'function',
    revision: plan.inheritedCodebaseRevision || undefined, source: plan.sourceSpan.file };
}

/** Whether a capture's declared type is a function type (natlang lambda text, or a host function contract). */
function functionTyped(target: TargetDescriptor, text: string): boolean {
  if (target.host?.kind === 'function') return true;
  try { const parsed = parseType(text); return parsed.kind === 'lambda'; } catch { return /^\s*\(.*\)\s*=>/.test(text); }
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

/** Actual parent eval identity; supplied by the tool executor, never inferred from text. */
export type InlineInstructionOrigin = { parentInvocationId: string; toolCallId: string; actionOrdinal: number;
  writtenCodeSha256: string; checkedCodeSha256: string; sourceTemplateSpan?: { start: number; end: number } };

function instructionSite(plan: InlineLambdaPlan, renderedValues: readonly string[], origin?: InlineInstructionOrigin) {
  return { schema: 'natlang.inline_instruction_site/1', definition_id: plan.definitionId,
    template_span: plan.softBody && origin?.sourceTemplateSpan ? origin.sourceTemplateSpan : plan.templateSpan,
    checked_template_span: plan.templateSpan, source_span: plan.sourceSpan, template_segments: plan.strings,
    interpolations: plan.interpolations.map((item, index) => ({ ...item, rendered: renderedValues[index] })),
    parameters: plan.parameters, returns: plan.returns, captures: plan.captures,
    ...(plan.softBody ? { soft_body_id: plan.softBody } : {}),
    ...(plan.explicitCaptures ? { explicit_captures: true } : {}), origin: origin ?? null };
}

function captureSnapshotAttestation(plan: InlineLambdaPlan, origin: InlineInstructionOrigin | undefined,
  snapshots: readonly { name: string; type: 'string' | 'number' | 'boolean'; declared_type?: string;
    source: 'local' | 'input' | 'block'; value: string | number | boolean }[]) {
  if (!origin || !origin.parentInvocationId || !origin.toolCallId || !Number.isSafeInteger(origin.actionOrdinal) ||
      origin.actionOrdinal < 0 || !/^[a-f0-9]{64}$/.test(origin.writtenCodeSha256) ||
      !/^[a-f0-9]{64}$/.test(origin.checkedCodeSha256) || snapshots.length === 0) return undefined;
  const site = instructionSite(plan, [], origin);
  const span = (value: unknown) => {
    if (!value || typeof value !== 'object') return undefined;
    const item = value as { start?: unknown; end?: unknown };
    return Number.isSafeInteger(item.start) && Number.isSafeInteger(item.end) && Number(item.start) >= 0 &&
      Number(item.end) > Number(item.start) ? value : undefined;
  };
  const sourceSpan = span(plan.sourceSpan), templateSpan = span(site.template_span), checkedTemplateSpan = span(site.checked_template_span);
  if (!sourceSpan || !templateSpan || !checkedTemplateSpan) return undefined;
  return { schema: 'natlang.runtime_capture_snapshots/1', captures: snapshots.map(capture => ({
    ...capture, mode: 'snapshot' as const,
    value_canonical: canonical({ type: capture.type, value: capture.value }),
    value_sha256: fingerprint({ type: capture.type, value: capture.value }, 'natlang.inline-capture-snapshot/v1'),
    creation: { parentInvocationId: origin.parentInvocationId, toolCallId: origin.toolCallId,
      actionOrdinal: origin.actionOrdinal, writtenCodeSha256: origin.writtenCodeSha256,
      checkedCodeSha256: origin.checkedCodeSha256, definitionId: plan.definitionId,
      sourceSpan, templateSpan, checkedTemplateSpan },
  })) };
}

export type CaptureAccessors = Record<string, readonly [() => unknown, ((value: unknown) => void)?]>;

/** Create an inline natlang callable instance for a compiled `nl` expression. */
export function inline(plan: InlineLambdaPlan, values: readonly unknown[], accessors: CaptureAccessors,
  context?: Record<string, unknown>, version: number = NATLANG_COMPILE_VERSION, bound?: import('./context.js').Frame, origin?: InlineInstructionOrigin): NatlangCallable {
  if (version !== NATLANG_COMPILE_VERSION)
    throw new Error(`this module was compiled for natlang output version ${version}; rebuild it with natlang build`);
  if (plan.explicitCaptures) return explicitInline(plan, values, accessors, context, bound, origin);
  const captures: Record<string, CaptureCell> = {};
  for (const capture of plan.captures) {
    const accessor = accessors[capture.name];
    if (!accessor) continue;
    const type = targetType(capture.type);
    // A function-typed binding is captured by value when the function is created: a live capture of a function
    // value is the one way a closure could reach itself (spec "Captures").
    if (functionTyped(capture.type, type)) {
      const value = accessor[0]();
      captures[capture.name] = { name: capture.name, type, mutable: false, get: () => value };
      continue;
    }
    captures[capture.name] = { name: capture.name, type, mutable: capture.mutable && !!accessor[1],
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
  return inlineCallable(planDefinition(plan, context), render, captures, undefined, bound,
    { inline_instruction_site: instructionSite(plan, renderedValues, origin) });
}

/**
 * `nl.with({ … })`: the captures are exactly the listed ones. Snapshot entries are read now, when the function is
 * created; `live(x)` entries are read at each call and written back. A template that is one Neuralese block is a soft
 * function: its body is that block, shown to the model as a literal, and the value is a callable `Neuralese<F>`.
 */
function explicitInline(plan: InlineLambdaPlan, values: readonly unknown[], accessors: CaptureAccessors,
  context: Record<string, unknown> | undefined, bound?: import('./context.js').Frame, origin?: InlineInstructionOrigin): NatlangCallable {
  const listed: Record<string, unknown> = {};
  const cells: Record<string, CaptureCell> = {};
  const snapshots: { name: string; type: 'string' | 'number' | 'boolean'; declared_type?: string;
    source: 'local' | 'input' | 'block'; value: string | number | boolean }[] = [];
  for (const capture of plan.captures) {
    const accessor = accessors[capture.name];
    if (!accessor) continue;
    const type = targetType(capture.type);
    if (capture.mode === 'live' && !functionTyped(capture.type, type)) {
      listed[capture.name] = live(accessor[0], accessor[1]);
      cells[capture.name] = { name: capture.name, type, mutable: !!accessor[1], get: accessor[0], ...(accessor[1] ? { set: accessor[1] } : {}) };
    } else {
      const value = accessor[0]();
      listed[capture.name] = value;
      cells[capture.name] = { name: capture.name, type, mutable: false, get: () => value };
      const descriptor = capture.type as { text?: unknown; natlang?: unknown };
      const primitiveType = typeof descriptor.natlang === 'string' ? descriptor.natlang :
        typeof descriptor.text === 'string' ? descriptor.text : '';
      const source = capture.source;
      const observedType = typeof value;
      const unknownDeclaredType = primitiveType === 'unknown' || primitiveType === 'any' ? primitiveType : undefined;
      const literalDescriptor = portablePrimitiveLiteral(primitiveType);
      const declaredLiteralType = !unknownDeclaredType && !['string', 'number', 'boolean'].includes(primitiveType) &&
        literalDescriptor?.type === observedType && Object.is(literalDescriptor.value, value) ? primitiveType : undefined;
      const portableObservedValue = observedType === 'string' || observedType === 'boolean' ||
        (observedType === 'number' && Number.isFinite(value) && !Object.is(value, -0));
      const matchesPortableDeclaredPrimitive = (primitiveType === 'string' || primitiveType === 'boolean' || primitiveType === 'number') &&
        observedType === primitiveType && (primitiveType !== 'number' || (Number.isFinite(value) && !Object.is(value, -0)));
      if (capture.mode === 'snapshot' && capture.mutable === false &&
          (source === 'local' || source === 'input' || source === 'block') &&
          (matchesPortableDeclaredPrimitive || unknownDeclaredType !== undefined && portableObservedValue || declaredLiteralType !== undefined))
        snapshots.push({ name: capture.name, type: observedType as 'string' | 'number' | 'boolean',
          ...(unknownDeclaredType ? { declared_type: unknownDeclaredType } : declaredLiteralType ? { declared_type: declaredLiteralType } : {}), source,
          value: value as string | number | boolean });
    }
  }
  const runtimeCaptureSnapshots = captureSnapshotAttestation(plan, origin, snapshots);
  if (plan.softBody) {
    const params = plan.parameters.map(parameter => `${parameter.name}: ${targetType(parameter.type)}`).join(', ');
    return softFunction({ type: `(${params}) => ${targetType(plan.returns)}`, body: plan.softBody, captures: listed,
      types: planTypes(plan), codebase: context ?? {}, name: `soft@${plan.sourceSpan.file.split('/').at(-1)}:${plan.sourceSpan.line}`,
      manifest: { inline_instruction_site: { ...instructionSite(plan, [], origin),
        ...(runtimeCaptureSnapshots ? { runtime_capture_snapshots: runtimeCaptureSnapshots } : {}) } } });
  }
  const renderedValues = values.map(interpolationText);
  const render = (frame: import('./context.js').Frame) => {
    const replacement = frame.task.programView.value(plan.definitionId, plan.programId);
    return interpolate(replacement?.kind === 'lambda.instructions' ? replacement.template.segments : plan.strings, renderedValues);
  };
  return inlineCallable(planDefinition(plan, context), render, cells, undefined, bound,
    { inline_instruction_site: { ...instructionSite(plan, renderedValues, origin),
      ...(runtimeCaptureSnapshots ? { runtime_capture_snapshots: runtimeCaptureSnapshots } : {}) } });
}

/** A named `.nl` import compiled into a module: the definition record embedded at build time. */
export function named(name: string, record: NatlangRecord): NatlangCallable {
  registerFileRecords(record);
  return namedCallable(name, record);
}

/** A callable folder such as `natlang.d/`, as a record of callables. */
export function folder(codebase: Record<string, ItemRecord>): Record<string, unknown> {
  registerFileRecords(codebase);
  return callableTree(codebase);
}

/**
 * A `.nz` import compiled into a module: the file is embedded at build time. Its blocks are kept for the runtime's
 * store (`importedBlocks`); soft-function exports become callables.
 */
export function nzModule(base64: string): Record<string, unknown> {
  const loaded = loadNzSync(fromBase64(base64));
  registerImportedBlocks(loaded.blocks.values());
  return nzExports(loaded);
}
export { importedBlocks };

/** Numeric loop bounds are fixed at entry; rounding must never stall the counter. */
export function numericProgress(initial: number, bound: number, upward: boolean): (current: number) => void {
  if (!Number.isFinite(initial) || !Number.isFinite(bound)) throw new RangeError('numeric loop requires a finite counter and bound');
  let previous: number | undefined;
  return current => {
    if (!Number.isFinite(current) || (previous !== undefined && !(upward ? current > previous : current < previous)))
      throw new RangeError('numeric loop counter did not advance toward its bound');
    previous = current;
  };
}

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
      const values = tag === '[object Map]' ? Array.from(Map.prototype.entries.call(source)) : Array.from(Set.prototype.values.call(source));
      const inner = (values as T[])[Symbol.iterator]();
      return { next: () => {
        if (collection.size > size) throw new RangeError('the collection grew while it was being iterated');
        return inner.next();
      } };
    } };
  }
  throw new TypeError(notIterable(source, label));
}

/**
 * Runtime half of the policy for `for await`: an async iterable passes, since restricted code cannot define one (it
 * comes from the host: a response body, a service stream, a package, an iterateOn stream); anything else follows the
 * `for ... of` rules.
 */
export function finiteAsync<T>(source: unknown, label?: string): AsyncIterable<T> | Iterable<T> {
  if (source && typeof source === 'object' && typeof (source as { [Symbol.asyncIterator]?: unknown })[Symbol.asyncIterator] === 'function')
    return source as AsyncIterable<T>;
  return finite(source as Iterable<T>, label);
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
