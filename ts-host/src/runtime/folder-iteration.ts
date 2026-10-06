/** Folder-owned iteration: every successful step publishes source and state together. */
import type { FolderHandle, FolderSnapshot } from '../native/scoped-fs.js';
import { Iteration, type IterationEvent, type ProgressJudgeFunction } from './iterate.js';

export type FolderIterationResult<S> = Readonly<{ folder: FolderSnapshot; state: S }>;
function immutableState<S>(value: S): S {
  // A checkpoint must not change when a later step mutates its own state, so plain objects and arrays are copied
  // and frozen. Everything else (functions, handles, snapshots, class instances) is kept by reference, and
  // undefined, cycles and shared substructure are preserved as they are.
  const copies = new Map<object, unknown>();
  const copy = (item: unknown): unknown => {
    if (item === null || typeof item !== 'object') return item;
    const plain = Array.isArray(item) || Object.getPrototypeOf(item) === Object.prototype || Object.getPrototypeOf(item) === null;
    if (!plain) return item;
    if (copies.has(item)) return copies.get(item);
    const target: Record<string | symbol, unknown> | unknown[] = Array.isArray(item) ? [] : Object.create(Object.getPrototypeOf(item));
    copies.set(item, target);
    for (const key of Reflect.ownKeys(item)) {
      const descriptor = Object.getOwnPropertyDescriptor(item, key)!;
      if ('value' in descriptor) descriptor.value = copy(descriptor.value);
      Object.defineProperty(target, key, descriptor);
    }
    return Object.freeze(target);
  };
  return copy(value) as S;
}
export class FolderIteration<S> {
  private readonly iteration: Iteration<FolderIterationResult<S>>;
  constructor(folder: FolderHandle, reducer: unknown, initial: S, fixed: unknown[]) {
    const checkpoint = Object.freeze({ folder: folder.snapshot(), state: immutableState(initial) });
    this.iteration = new Iteration(async previous => {
      const draft = previous.folder.branch();
      const state = immutableState(await draft.apply(reducer, previous.state, ...fixed) as S);
      return Object.freeze({ folder: draft.snapshot(), state });
    }, checkpoint, []);
  }
  until(done: (state: S, folder: FolderSnapshot) => boolean | Promise<boolean>): Promise<FolderIterationResult<S>> {
    return this.iteration.until(checkpoint => done(checkpoint.state, checkpoint.folder));
  }
  streamUntil(done: (state: S, folder: FolderSnapshot) => boolean | Promise<boolean>) {
    return this.iteration.streamUntil(checkpoint => done(checkpoint.state, checkpoint.folder));
  }
  onStep(observer: (event: IterationEvent<FolderIterationResult<S>>) => void | Promise<void>): this { this.iteration.onStep(observer); return this; }
  checkProgress(judge: ProgressJudgeFunction | 'off'): this { this.iteration.checkProgress(judge); return this; }
  withMeasure(remaining: (state: S, folder: FolderSnapshot) => number): this {
    this.iteration.withMeasure(checkpoint => remaining(checkpoint.state, checkpoint.folder)); return this;
  }
  withLimit(limit: { maxSteps?: number; deadlineMs?: number }): this { this.iteration.withLimit(limit); return this; }
  withSiteId(id: string): this { this.iteration.withSiteId(id); return this; }
  withCompilerSite(id: string): this { this.iteration.withCompilerSite(id); return this; }
  inFrame(frame: import('./context.js').Frame | undefined): this { this.iteration.inFrame(frame); return this; }
}
