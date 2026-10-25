/**
 * An asynchronous evaluator cannot expose a guest getter as a synchronous host
 * function. Read the guest cells at the native call boundary, run the ordinary
 * Natlang kernel against those cells, then commit its writes by selected calls.
 */
import type { InlineLambdaPlan } from '../compiler/inline.js';
import { selectedNativeIdentity } from '../native/selected-surface.js';
import { callableMeta, makeCallable, type NatlangCallable } from './callable.js';

type Accessor = readonly [() => unknown | Promise<unknown>, ((value: unknown) => unknown | Promise<unknown>)?];
type Accessors = Record<string, Accessor>;

function copyGraph(value: unknown, seen = new Map<object, unknown>()): unknown {
  if (!value || typeof value !== 'object') return value;
  // Resource state is owned by its native capability. Only replacement of the
  // captured handle is a cell write; copying private slots would invent a broken
  // object and confuse a native resource mutation with a guest binding update.
  if (selectedNativeIdentity(value)) return value;
  if (seen.has(value)) return seen.get(value);
  const result: Record<PropertyKey, unknown> | unknown[] = Array.isArray(value) ? [] : Object.create(Object.getPrototypeOf(value));
  seen.set(value, result);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!('value' in descriptor)) throw new TypeError('Selected capture snapshots require data properties');
    if (Array.isArray(result) && key === 'length') { result.length = descriptor.value; continue; }
    Object.defineProperty(result, key, { value: copyGraph(descriptor.value, seen), enumerable: descriptor.enumerable,
      writable: true, configurable: true });
  }
  return result;
}

// Compare snapshots without getters, JSON coercion, or loss of NaN/-0/aliases.
function sameGraph(a: unknown, b: unknown, seen = new Map<object, object>(), reverse = new Map<object, object>()): boolean {
  if (Object.is(a, b)) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  if (selectedNativeIdentity(a) || selectedNativeIdentity(b)) return false;
  if (seen.has(a)) return seen.get(a) === b;
  if (reverse.has(b)) return false;
  seen.set(a, b); reverse.set(b, a);
  const left = Object.getOwnPropertyDescriptors(a), right = Object.getOwnPropertyDescriptors(b);
  const keys = Reflect.ownKeys(left);
  if (keys.length !== Reflect.ownKeys(right).length) return false;
  return keys.every(key => {
    const x = Reflect.get(left, key), y = Reflect.get(right, key);
    return x && y && 'value' in x && 'value' in y && sameGraph(x.value, y.value, seen, reverse);
  });
}

/** The factory is a declared native compiler helper; its callback bodies stay selected. */
export async function withSelectedCaptureAccessors(plan: InlineLambdaPlan, accessors: Record<string, unknown>,
    create: (accessors: Accessors) => unknown): Promise<NatlangCallable> {
  const cells = new Map<string, { accessor: Accessor; value: unknown; baseline: unknown; live: boolean; writable: boolean }>();
  const bound: Accessors = {};
  for (const capture of plan.captures) {
    const accessor = accessors[capture.name] as Accessor | undefined;
    if (!accessor) continue;
    if (!Array.isArray(accessor) || typeof accessor[0] !== 'function' ||
        (accessor[1] !== undefined && typeof accessor[1] !== 'function'))
      throw new TypeError(`selected capture ${capture.name} requires an owned getter and optional setter`);
    if (capture.shadowedByParameter && capture.mode === 'live') continue;
    const type = capture.type as { kind?: string; text?: string; natlang?: string };
    const functionTyped = type.kind === 'function' || /=>|^\s*\(/.test(type.natlang ?? type.text ?? '');
    const live = !functionTyped && !capture.shadowedByParameter && (!plan.explicitCaptures || capture.mode === 'live');
    const value = await accessor[0]();
    const cell = { accessor, value, baseline: copyGraph(value), live, writable: live && !!accessor[1] && capture.mutable };
    cells.set(capture.name, cell);
    bound[capture.name] = cell.writable ? [() => cell.value, (next: unknown) => { cell.value = next; }] : [() => cell.value];
  }
  const callable = create(bound), meta = callableMeta(callable);
  if (!meta) throw new TypeError('The selected inline factory must return a Natlang callable');
  let tail: Promise<unknown> = Promise.resolve();
  const wrapped = makeCallable({ ...meta, invoke: (args, frame, at) => {
    // Two calls sharing these cells cannot observe each other's half-committed
    // writes. Work on unrelated callables remains independently schedulable.
    const work = tail.then(async () => {
      for (const cell of cells.values()) if (cell.live) {
        cell.value = await cell.accessor[0](); cell.baseline = copyGraph(cell.value);
      }
      const result = await meta.invoke(args, frame, at);
      const writes = [...cells.values()].filter(cell => cell.writable && !sameGraph(cell.value, cell.baseline));
      // Validate every captured binding before admitting the first write.
      for (const cell of writes) if (!sameGraph(await cell.accessor[0](), cell.baseline))
        throw new Error('capture-conflict: a selected capture changed during its native child call');
      for (const cell of writes) await cell.accessor[1]!(cell.value);
      return result;
    });
    tail = work.catch(() => {});
    return work;
  } });
  return wrapped;
}
