/**
 * Task context and its propagation.
 *
 * A frame identifies the task a natlang call belongs to and the chain of natlang definitions
 * that led to it. Node propagates frames with `AsyncLocalStorage`; browsers use a synchronous
 * slot that compiled code restores after every `await` (see `bindAwait`).
 */
import type { NatlangTask } from './runtime.js';
import type { FileHandle, Folder, FolderHandle } from '../native/scoped-fs.js';

/** Root/named calls start at zero; five nested ad hoc calls may be active below them. */
export const MAX_AD_HOC_NL_DEPTH = 5;
export const canGenerateNl = (frame?: Frame): boolean => (frame?.adHocDepth ?? 0) < MAX_AD_HOC_NL_DEPTH;

export type Frame = Readonly<{
  task: NatlangTask;
  /** Call identities of the natural-language definitions that are currently active above this point. */
  chain: readonly string[];
  /** The authored TypeScript functions currently active above this point, innermost first. */
  active?: ActiveCall;
  /** Call ID of the natlang invocation that created this frame, if any. */
  parentCallId?: string;
  programOwner?: string;
  /** Whether that invocation is an inline `nl` function: a judgment eval code handed over. */
  inline?: boolean;
  /** Active ad hoc nl/delegate layers since the most recent file-backed .nl root. */
  adHocDepth?: number;
  /** System prompt text for the one invocation started in this frame (not inherited by its children). */
  systemAddendum?: string;
  /** Ancestor-visible scoped handles rebased into this invocation's copy-on-write view. */
  scopedHandleReplacements?: ReadonlyMap<Folder | FolderHandle | FileHandle, Folder | FolderHandle | FileHandle>;
  /**
   * Aborts when the work running in this frame should stop: the task is cancelled, the call this frame belongs to
   * failed, or the eval that started it failed or finished without awaiting it. Child calls derive theirs from it.
   */
  signal?: AbortSignal;
  /** The controller of the natural-language call this frame belongs to (aborts its remaining children when it fails). */
  abort?: AbortController;
}>;

/** One active call of an authored TypeScript function, linked to the calls active above it. */
export type ActiveCall = {
  readonly id: string;
  readonly args: readonly unknown[];
  readonly parent?: ActiveCall;
  /** The nearest call above this one of the same function. */
  readonly same?: ActiveCall;
  /** Argument positions that got smaller at every re-entry of this function since its first call in the chain. */
  readonly descent?: readonly Descent[];
  /** Direct parts of an argument, collected the first time a re-entry is checked against it. */
  parts?: Map<number, Set<unknown>>;
};
type Descent = { readonly position: number; readonly kind: 'part' | 'length' | 'integer' };

/**
 * Natural-language call promises that eval code passed to Promise.race or Promise.any. A raced call still running when
 * its eval finishes lost the race: it is stopped without failing the eval.
 */
export const racedCalls = new WeakSet<object>();
/** The kernel's tracked call promise behind each promise a natlang callable returned. */
export const startedCalls = new WeakMap<object, object>();
/** Record that eval code raced these values (Promise.race, Promise.any). */
export function markRaced(values: readonly unknown[]): void {
  for (const value of values) if (value && (typeof value === 'object' || typeof value === 'function')) {
    racedCalls.add(value);
    const started = startedCalls.get(value);
    if (started) racedCalls.add(started);
  }
}

/**
 * Promise.race and Promise.any in an eval realm record the calls they were given: a natural-language call that lost
 * the race and is still running when its eval finishes is stopped without failing the eval (spec: Eval).
 */
export function markRaces(realmPromise: PromiseConstructor): void {
  for (const name of ['race', 'any'] as const) {
    const original = realmPromise[name] as (this: PromiseConstructor, values: unknown[]) => Promise<unknown>;
    Object.defineProperty(realmPromise, name, { configurable: true, writable: true, value: {
      [name](this: PromiseConstructor, values: Iterable<unknown>) {
        const items = Array.from(values);
        markRaced(items);
        return original.call(this, items);
      } }[name] });
  }
}

export interface ContextStore {
  current(): Frame | undefined;
  run<T>(frame: Frame | undefined, fn: () => T): T;
}

/** Synchronous slot. Compiled browser code restores it after each `await`. */
export class SlotContextStore implements ContextStore {
  private active?: Frame;
  current(): Frame | undefined { return this.active; }
  run<T>(frame: Frame | undefined, fn: () => T): T {
    const previous = this.active;
    this.active = frame;
    try { return fn(); } finally { this.active = previous; }
  }
  /** Used by the await-restoration transform. */
  restore(frame: Frame | undefined): void { this.active = frame; }
}

let store: ContextStore = new SlotContextStore();

/** Install the platform context store (Node installs `AsyncLocalStorage`). */
export function setContextStore(next: ContextStore): void { store = next; }
export function contextStore(): ContextStore { return store; }
export function currentFrame(): Frame | undefined { return store.current(); }
export function runInFrame<T>(frame: Frame | undefined, fn: () => T): T { return store.run(frame, fn); }

/**
 * Browsers have no AsyncLocalStorage: a callback a promise or timer runs later would lose the task frame, and with it
 * the recursion guard's view of what is running. While a frame is current, these hand it to the callbacks they
 * schedule, as AsyncLocalStorage does in Node; with none current they behave as before. Installed once per realm.
 */
export function propagateSlotFrames(slot: SlotContextStore, realm: typeof globalThis = globalThis): void {
  const marker = Symbol.for('natlang.slotFrames');
  const target = realm as unknown as Record<symbol, unknown>;
  if (target[marker]) return;
  target[marker] = true;
  const carry = <F>(callback: F, frame: Frame): F => typeof callback !== 'function' ? callback :
    (function (this: unknown, ...args: unknown[]) {
      const previous = slot.current();
      slot.restore(frame);
      try { return (callback as (...items: unknown[]) => unknown).apply(this, args); } finally { slot.restore(previous); }
    }) as F;
  const then = realm.Promise.prototype.then;
  Object.defineProperty(realm.Promise.prototype, 'then', { configurable: true, writable: true,
    value: function (this: Promise<unknown>, fulfilled?: unknown, rejected?: unknown) {
      const frame = slot.current();
      return frame ? then.call(this, carry(fulfilled, frame) as never, carry(rejected, frame) as never) :
        then.call(this, fulfilled as never, rejected as never);
    } });
  for (const name of ['setTimeout', 'queueMicrotask', 'requestAnimationFrame'] as const) {
    const original = (realm as unknown as Record<string, unknown>)[name];
    if (typeof original !== 'function') continue;
    (realm as unknown as Record<string, unknown>)[name] = function (this: unknown, callback: unknown, ...rest: unknown[]) {
      const frame = slot.current();
      return (original as (...items: unknown[]) => unknown).call(this, frame ? carry(callback, frame) : callback, ...rest);
    };
  }
}

/**
 * Await-restoration helper emitted by the browser transform: `await x` becomes
 * `(await bindAwait(x))()`. The frame is captured before suspension and restored on resumption.
 */
export function bindAwait<T>(value: T | PromiseLike<T>): Promise<() => T> {
  const saved = store.current();
  const slot = store instanceof SlotContextStore ? store : undefined;
  return Promise.resolve(value).then(
    resolved => () => { slot?.restore(saved); return resolved; },
    error => () => { slot?.restore(saved); throw error; });
}

export class NatlangContextError extends Error {
  constructor(message = 'No natlang task is active here. Call natlang functions inside runtime.run(...), ' +
    'or wrap a callback that runs later with runtime.bind(fn).') { super(message); this.name = 'NatlangContextError'; }
}

export class NatlangRecursionError extends Error {
  constructor(readonly definitionId: string, readonly chain: readonly string[], label = definitionId, message?: string) {
    super(message ?? `${label} was called while it is already running in its own call chain; recursion is not allowed in natlang callable code.`);
    this.name = 'NatlangRecursionError';
  }
}

/** The readable name in an authored callable's guard ID (`prefix#name@position`). */
export const guardName = (id: string): string => /#([^#@]+)@\d+$/.exec(id)?.[1] ?? id;

/**
 * Enter an authored function. A function already active in its own call chain may run again only on a smaller
 * argument: a part of the argument it had (reachable through its properties or elements), a shorter array or string,
 * or a smaller non-negative integer. The same position must keep getting smaller along the chain, and a part may not
 * repeat, so every chain of re-entries ends (spec: Iteration and termination).
 */
export function guard<T>(id: string, fn: () => T, args: readonly unknown[] = [], label?: string): T {
  const frame = store.current();
  if (!frame) return fn();
  let same = frame.active;
  while (same && same.id !== id) same = same.parent;
  let descent: Descent[] | undefined;
  if (same) {
    descent = descend(same, args);
    if (!descent.length) {
      const name = label ?? guardName(id);
      throw new NatlangRecursionError(id, frame.chain, name, `\`${name}\` called itself without a smaller argument. A function ` +
        'may call itself only on a smaller argument: a part of its input, a shorter array or string, or a smaller ' +
        'non-negative integer. Otherwise use a loop over a work list, or iterateOn.');
    }
  }
  return store.run({ ...frame, active: { id, args, parent: frame.active, same, descent } }, fn);
}

/** The argument positions on which a re-entry is smaller than `previous`, among those still decreasing. */
function descend(previous: ActiveCall, args: readonly unknown[]): Descent[] {
  const candidates: readonly Descent[] = previous.descent ?? Array.from({ length: Math.min(previous.args.length, args.length) },
    (_, position) => (['part', 'length', 'integer'] as const).map(kind => ({ position, kind }))).flat();
  return candidates.filter(({ position, kind }) => {
    const before = previous.args[position], now = args[position];
    if (kind === 'integer') return Number.isSafeInteger(before) && Number.isSafeInteger(now) && (now as number) >= 0 &&
      (now as number) < (before as number);
    if (kind === 'length') return (Array.isArray(before) && Array.isArray(now)) || (typeof before === 'string' && typeof now === 'string') ?
      (now as { length: number }).length < (before as { length: number }).length : false;
    // A part may be a primitive leaf (which has no parts of its own, so the chain ends there); an object part may not
    // repeat, which keeps a cyclic structure from being walked round and round.
    if (!isObject(before) || now === before || !partOf(previous, position, now)) return false;
    if (isObject(now)) for (let entry: ActiveCall | undefined = previous; entry; entry = entry.same) if (entry.args[position] === now) return false;
    return true;
  });
}

const isObject = (value: unknown): value is object => !!value && (typeof value === 'object' || typeof value === 'function');

/** The values directly inside a value: array elements, own enumerable properties, Map and Set entries. */
function partsOf(value: object): unknown[] {
  if (value instanceof Map) return [...value.keys(), ...value.values()];
  if (value instanceof Set) return [...value];
  return Array.isArray(value) ? value : Object.values(value);
}

/** Whether `part` is reachable from the argument at `position` of an active call, checking its direct parts first. */
function partOf(call: ActiveCall, position: number, part: unknown): boolean {
  const whole = call.args[position] as object;
  call.parts ??= new Map();
  let direct = call.parts.get(position);
  if (!direct) call.parts.set(position, direct = new Set(partsOf(whole)));
  if (direct.has(part)) return true;
  const seen = new Set<object>([whole]);
  let frontier = [...direct].filter(isObject);
  while (frontier.length) {
    const next: object[] = [];
    for (const item of frontier) {
      if (seen.has(item)) continue;
      seen.add(item);
      for (const inner of partsOf(item)) {
        if (inner === part) return true;
        if (isObject(inner) && !seen.has(inner)) next.push(inner);
      }
    }
    frontier = next;
  }
  return false;
}
