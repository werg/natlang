/** Folder-owned iteration: every successful step publishes source and state together. */
import type { FolderHandle, FolderSnapshot } from '../native/scoped-fs.js';
import { Iteration, type IterationEvent, type ProgressJudgeFunction } from './iterate.js';

export type FolderIterationResult<S> = Readonly<{ folder: FolderSnapshot; state: S }>;
function immutableState<S>(value: S): S {
  // State is portable JSON. Capabilities belong in fixed arguments, never in checkpoints.
  const visit = (item: unknown, seen: Set<object>, path: string): void => {
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return;
    if (typeof item === 'number' && Number.isFinite(item)) return;
    if (typeof item !== 'object' || seen.has(item)) throw new TypeError(`folder iteration state must be finite, acyclic JSON at ${path}; received ${typeof item}`);
    if (!Array.isArray(item) && Object.getPrototypeOf(item) !== Object.prototype) throw new TypeError(`folder iteration state must be portable JSON at ${path}`);
    seen.add(item);
    if (Array.isArray(item)) {
      for (let index = 0; index < item.length; index++) visit(item[index], seen, `${path}[${index}]`);
    } else {
      for (const [key, child] of Object.entries(item)) visit(child, seen, `${path}[${JSON.stringify(key)}]`);
    }
    seen.delete(item);
  };
  visit(value, new Set(), '$');
  const copy = JSON.parse(JSON.stringify(value)) as S;
  const freeze = (item: unknown): void => { if (item && typeof item === 'object') { Object.values(item).forEach(freeze); Object.freeze(item); } };
  freeze(copy); return copy;
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
