/**
 * Task context and its propagation.
 *
 * A frame identifies the task a natlang call belongs to and the chain of natlang definitions
 * that led to it. Node propagates frames with `AsyncLocalStorage`; browsers use a synchronous
 * slot that compiled code restores after every `await` (see `bindAwait`).
 */
import type { NatlangTask } from './runtime.js';

export type Frame = Readonly<{
  task: NatlangTask;
  /** Definition IDs of the natlang or authored callables that are currently active above this point. */
  chain: readonly string[];
  /** Call ID of the natlang invocation that created this frame, if any. */
  parentCallId?: string;
}>;

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
  constructor(readonly definitionId: string, readonly chain: readonly string[], label = definitionId) {
    super(`${label} was called while it is already running in its own call chain; recursion is not allowed in natlang callable code.`);
    this.name = 'NatlangRecursionError';
  }
}

/** Enter a callable: reject reentry in its own caller chain, then run `fn` in the extended frame. */
export function guard<T>(id: string, fn: () => T, label?: string): T {
  const frame = store.current();
  if (!frame) return fn();
  if (frame.chain.includes(id)) throw new NatlangRecursionError(id, frame.chain, label);
  return store.run({ ...frame, chain: [...frame.chain, id] }, fn);
}
