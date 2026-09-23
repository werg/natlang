/**
 * `iterateOn`: the sanctioned open-ended iteration operator.
 *
 * The step is called sequentially with the current state and fixed arguments; its result is the
 * next state. The stopping predicate is checked on the initial state and after every step. Progress
 * reviews are scheduled from per-site statistics and never stop a run by themselves: only a
 * `divergent` verdict does.
 */
import { hexDigest } from '../native/hash.js';
import { isLive, liveId } from '../native/values.js';
import { currentFrame, runInFrame } from './context.js';
import { callableMeta } from './callable.js';
import { invokeDefinition, type CallableDefinition } from './kernel.js';
import { resolveFrame } from './runtime.js';

export type IterationEvent<T> = Readonly<{
  kind: 'initial' | 'step' | 'review' | 'done' | 'error';
  sequence: number; iteration: number; iterationId: string; siteId: string;
  state?: T; review?: ProgressVerdict; error?: string;
}>;
export type ProgressVerdict = { verdict: 'continue' | 'divergent'; reason: string };
export type StepRecord = { iteration: number; callId: string; elapsedMs: number; stateHash: string };

export interface IterationTrajectory<T> {
  readonly iterationId: string;
  readonly siteId: string;
  readonly length: number;
  state(index: number): T;
  states(start?: number, end?: number): T[];
  steps(start?: number, end?: number): StepRecord[];
  repeats(): { iteration: number; firstSeen: number }[];
  summary(): { iterations: number; activeMs: number; repeatedStates: number; reviews: number; statistics: unknown };
}
export type ProgressJudgeFunction = <T>(trajectory: IterationTrajectory<T>, context: JudgeContext) => Promise<ProgressVerdict>;
export type JudgeContext = { stepName: string; stepInstructions?: string; predicateInstructions?: string; stateType?: string };

export type SiteStatistics = { successes: number; meanSteps: number; m2Steps: number; meanMs: number; m2Ms: number;
  failures: number; wallMs: number };
export interface IterationStatisticsStore {
  read(key: string): SiteStatistics | undefined | Promise<SiteStatistics | undefined>;
  write(key: string, value: SiteStatistics): void | Promise<void>;
  reset?(prefix?: string): void | Promise<void>;
  export?(): Record<string, SiteStatistics> | Promise<Record<string, SiteStatistics>>;
}

/** Default statistics store: in memory for the lifetime of the process. */
export class MemoryIterationStatistics implements IterationStatisticsStore {
  private readonly values = new Map<string, SiteStatistics>();
  read(key: string) { return this.values.get(key); }
  write(key: string, value: SiteStatistics) { this.values.set(key, value); }
  reset(prefix = '') { for (const key of [...this.values.keys()]) if (key.startsWith(prefix)) this.values.delete(key); }
  export() { return Object.fromEntries(this.values); }
}
const defaultStatistics = new MemoryIterationStatistics();

class IterationFailure extends Error {
  constructor(message: string, readonly lastState: unknown, readonly trajectory: IterationTrajectory<unknown>, options?: ErrorOptions) {
    super(message, options);
  }
}
/** A progress review judged the run divergent. */
export class IterationDivergedError extends IterationFailure {
  constructor(readonly reason: string, lastState: unknown, trajectory: IterationTrajectory<unknown>) {
    super(`iteration diverged: ${reason}`, lastState, trajectory); this.name = 'IterationDivergedError';
  }
}
/** A step or stopping predicate failed; the last checked state and trajectory are attached. */
export class IterationStepError extends IterationFailure {
  constructor(message: string, lastState: unknown, trajectory: IterationTrajectory<unknown>, cause: unknown) {
    super(message, lastState, trajectory, { cause }); this.name = 'IterationStepError';
  }
}
/** A caller-supplied `withLimit` bound was reached. */
export class IterationLimitError extends IterationFailure {
  constructor(message: string, lastState: unknown, trajectory: IterationTrajectory<unknown>) {
    super(message, lastState, trajectory); this.name = 'IterationLimitError';
  }
}

function stableHash(value: unknown): string {
  const seen = new WeakSet<object>();
  const canonical = (item: unknown): unknown => {
    if (item === null || typeof item !== 'object') return typeof item === 'function' ? `live:${liveId(item)}` : item;
    if (isLive(item)) return `live:${liveId(item)}`;
    if (seen.has(item)) return '[cycle]';
    seen.add(item);
    if (Array.isArray(item)) return item.map(canonical);
    return Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => [k, canonical(v)]));
  };
  return hexDigest(JSON.stringify(canonical(value)) ?? 'undefined').slice(0, 16);
}

type Step<T> = (state: T, ...args: unknown[]) => T | Promise<T>;
type Done<T> = (state: T) => boolean | Promise<boolean>;

/** Decide whether a completed step should be reviewed. Bootstrap sites are reviewed occasionally. */
export function reviewDue(stats: SiteStatistics | undefined, steps: number, activeMs: number, lastReviewAt: number): boolean {
  if (steps < 3) return false;
  const since = steps - lastReviewAt;
  if (!stats || stats.successes < 5) return steps >= 10 && (lastReviewAt === 0 ? true : since >= Math.max(10, lastReviewAt));
  const band = (value: number, mean: number, m2: number, floor: number) => {
    const sd = Math.max(Math.sqrt(m2 / Math.max(1, stats.successes - 1)), floor, 0.25 * mean);
    return (value - mean) / sd;
  };
  const z = Math.max(band(steps, stats.meanSteps, stats.m2Steps, 1), band(activeMs, stats.meanMs, stats.m2Ms, 1000));
  const every = z >= 3 ? 1 : z >= 2 ? 2 : z >= 1 ? 4 : Infinity;
  return since >= every;
}

function updateStatistics(previous: SiteStatistics | undefined, steps: number, activeMs: number, wallMs: number,
  success: boolean): SiteStatistics {
  const stats = previous ? { ...previous } : { successes: 0, meanSteps: 0, m2Steps: 0, meanMs: 0, m2Ms: 0, failures: 0, wallMs: 0 };
  if (!success) { stats.failures++; return stats; }
  stats.successes++;
  const dSteps = steps - stats.meanSteps; stats.meanSteps += dSteps / stats.successes; stats.m2Steps += dSteps * (steps - stats.meanSteps);
  const dMs = activeMs - stats.meanMs; stats.meanMs += dMs / stats.successes; stats.m2Ms += dMs * (activeMs - stats.meanMs);
  stats.wallMs = wallMs;
  return stats;
}

const DEFAULT_JUDGE_INSTRUCTIONS = `An iterative process has been running for a while. Decide whether it is still making meaningful progress.
Inspect the trajectory: use trajectory.summary(), trajectory.steps(), trajectory.repeats(), and trajectory.states(start, end) to page through states.
Distinguish real improvement from repetition, oscillation between the same few states, unproductive churn, and a goal that looks impossible.
An unusually long run that is still improving should continue.
Return { verdict: "continue", reason } if further steps are likely to help, or { verdict: "divergent", reason } if the process is stuck or cannot succeed. Give a concrete reason.`;

/** The library progress judge: a natlang lambda with read-only access to the whole trajectory. */
export const defaultProgressJudge: ProgressJudgeFunction = async (trajectory, context) => {
  const frame = currentFrame() ?? resolveFrame();
  const definition: CallableDefinition = { id: 'natlang:progress_judge', name: 'progress_judge',
    description: 'Library progress judge for iterateOn.', body: DEFAULT_JUDGE_INSTRUCTIONS,
    params: [{ name: 'trajectory', type: 'Live<"IterationTrajectory", "shape", "summary,states,steps,repeats">' },
      { name: 'step', type: 'string' }],
    returns: 'ProgressVerdict', types: { ProgressVerdict: '{ verdict: "continue" | "divergent", reason: string }' },
    codebase: {}, subtype: 'function' };
  const step = `${context.stepName}${context.stepInstructions ? `: ${context.stepInstructions}` : ''}` +
    (context.predicateInstructions ? `\nStop when: ${context.predicateInstructions}` : '');
  return await invokeDefinition(frame, definition, [trajectory, step]) as ProgressVerdict;
};

export class Iteration<T> {
  private observers: ((event: IterationEvent<T>) => void | Promise<void>)[] = [];
  private judge?: ProgressJudgeFunction;
  private siteId?: string;
  private compilerSite?: string;
  private limit: { maxSteps?: number; deadlineMs?: number } = {};
  private started = false;
  private frame?: import('./context.js').Frame;
  readonly [Symbol.toStringTag] = 'Iteration';

  constructor(private readonly step: Step<T>, private readonly initial: T, private readonly fixed: unknown[]) {
    if (typeof step !== 'function') throw new TypeError('iterateOn requires a step function');
  }

  onStep(observer: (event: IterationEvent<T>) => void | Promise<void>): this { this.observers.push(observer); return this; }
  checkProgress(judge: ProgressJudgeFunction): this { this.judge = judge; return this; }
  withSiteId(id: string): this {
    if (!id.trim()) throw new TypeError('withSiteId requires a non-empty identifier');
    this.siteId = id; return this;
  }
  withLimit(limit: { maxSteps?: number; deadlineMs?: number }): this { this.limit = { ...limit }; return this; }
  /** Run in a fixed task frame (used when an interpreter session creates the iteration). */
  inFrame(frame: import('./context.js').Frame | undefined): this { this.frame ??= frame; return this; }
  /** Set by compiled code with the call site's stable identity. */
  withCompilerSite(id: string): this { this.compilerSite ??= id; return this; }

  until(done: Done<T>): Promise<T> { return this.run(done, async () => {}); }

  streamUntil(done: Done<T>): AsyncIterable<IterationEvent<T>> & { readonly __natlangIterationStream: true } {
    const queue: IterationEvent<T>[] = [];
    let wake: (() => void) | undefined, finished = false, failure: unknown, cancelled = false;
    let running: Promise<T> | undefined;
    const start = () => {
      running ??= this.run(done, async event => {
        queue.push(event); wake?.();
        if (cancelled) throw new Error('iteration stream consumer stopped');
      }).then(value => { finished = true; wake?.(); return value; },
        error => { finished = true; failure = error; wake?.(); throw error; });
      running.catch(() => {});
    };
    return { __natlangIterationStream: true, [Symbol.asyncIterator]: () => ({
      next: async (): Promise<IteratorResult<IterationEvent<T>>> => {
        start();
        while (!queue.length && !finished) await new Promise<void>(resolve => { wake = resolve; });
        wake = undefined;
        if (queue.length) return { value: queue.shift()!, done: false };
        if (failure && !cancelled) throw failure;
        return { value: undefined, done: true };
      },
      return: async (): Promise<IteratorResult<IterationEvent<T>>> => { cancelled = true; return { value: undefined, done: true }; },
    }) };
  }

  private async run(done: Done<T>, emitStream: (event: IterationEvent<T>) => Promise<void>): Promise<T> {
    if (this.started) throw new Error('an Iteration can be started only once; call iterateOn again for another run');
    this.started = true;
    const frame = this.frame ?? currentFrame() ?? resolveFrame();
    const task = frame.task;
    const iterationId = `iter-${Math.random().toString(36).slice(2, 10)}`;
    const stepMeta = callableMeta(this.step), doneMeta = callableMeta(done);
    const stepId = stepMeta?.definition.id ?? (this.step.name || 'step');
    const siteId = this.siteId ?? this.compilerSite ?? `unsited:${stepId}`;
    const persistent = !!(this.siteId ?? this.compilerSite);
    const store = task.runtime.options.statistics ?? defaultStatistics;
    const statsKey = `${siteId}|${stepId}|${doneMeta?.definition.id ?? 'predicate'}`;
    const stats = persistent ? await store.read(statsKey) : undefined;
    const judge = this.judge ?? task.runtime.options.progressJudge ?? defaultProgressJudge;
    const states: T[] = [this.initial];
    const steps: StepRecord[] = [];
    const hashes = new Map<string, number>([[stableHash(this.initial), 0]]);
    const repeats: { iteration: number; firstSeen: number }[] = [];
    let reviews = 0, activeMs = 0, sequence = 0, lastReviewAt = 0;
    const started = Date.now();
    const trajectory: IterationTrajectory<T> = {
      iterationId, siteId,
      get length() { return states.length; },
      state: index => { if (!Number.isInteger(index) || index < 0 || index >= states.length) throw new RangeError('no such state'); return states[index]!; },
      states: (start = 0, end = states.length) => states.slice(start, Math.min(end, start + 50)),
      steps: (start = 0, end = steps.length) => steps.slice(start, Math.min(end, start + 200)),
      repeats: () => [...repeats],
      summary: () => ({ iterations: steps.length, activeMs, repeatedStates: repeats.length, reviews,
        statistics: stats ? { successfulRuns: stats.successes, meanSteps: Math.round(stats.meanSteps * 10) / 10,
          meanActiveMs: Math.round(stats.meanMs) } : 'no history for this site' }),
    };
    const emit = async (event: Omit<IterationEvent<T>, 'sequence' | 'iterationId' | 'siteId'>) => {
      const full = Object.freeze({ ...event, sequence: sequence++, iterationId, siteId }) as IterationEvent<T>;
      for (const observer of this.observers) await observer(full);
      await emitStream(full);
      traceEvents.push({ kind: full.kind, sequence: full.sequence, iteration: full.iteration,
        state_hash: full.state === undefined ? null : stableHash(full.state), review: full.review ?? null, error: full.error ?? null });
    };
    const traceEvents: Record<string, unknown>[] = [];
    let state = this.initial;
    let outcome = 'done';
    const check = async (): Promise<boolean> => {
      try { return await runInFrame(frame, () => done(state)) === true; }
      catch (error) { throw new IterationStepError(`stopping predicate failed: ${(error as Error)?.message ?? error}`, state, trajectory as never, error); }
    };
    try {
      await emit({ kind: 'initial', iteration: 0, state });
      if (await check()) { await emit({ kind: 'done', iteration: 0, state }); return state; }
      while (true) {
        task.checkOpen();
        if (this.limit.maxSteps !== undefined && steps.length >= this.limit.maxSteps)
          throw new IterationLimitError(`iteration reached its limit of ${this.limit.maxSteps} steps`, state, trajectory as never);
        if (this.limit.deadlineMs !== undefined && Date.now() - started >= this.limit.deadlineMs)
          throw new IterationLimitError(`iteration reached its deadline of ${this.limit.deadlineMs} ms`, state, trajectory as never);
        const before = Date.now();
        let next: T;
        try { next = await runInFrame(frame, () => this.step(state, ...this.fixed)); }
        catch (error) { throw new IterationStepError(`step ${steps.length + 1} failed: ${(error as Error)?.message ?? error}`, state, trajectory as never, error); }
        const elapsed = Date.now() - before;
        activeMs += elapsed;
        state = next;
        states.push(state);
        const hash = stableHash(state);
        const iteration = steps.length + 1;
        if (hashes.has(hash)) repeats.push({ iteration, firstSeen: hashes.get(hash)! }); else hashes.set(hash, iteration);
        steps.push({ iteration, callId: task.traces.at(-1)?.callId ?? '', elapsedMs: elapsed, stateHash: hash });
        await emit({ kind: 'step', iteration, state });
        if (await check()) { await emit({ kind: 'done', iteration, state }); return state; }
        if (reviewDue(stats, iteration, activeMs, lastReviewAt)) {
          lastReviewAt = iteration; reviews++;
          const verdict = await runInFrame(frame, () => judge(trajectory, { stepName: stepMeta?.definition.name ?? (this.step.name || 'step'),
            stepInstructions: stepMeta?.definition.body.trim(), predicateInstructions: doneMeta?.definition.body.trim() }));
          await emit({ kind: 'review', iteration, review: verdict });
          if (verdict?.verdict === 'divergent') throw new IterationDivergedError(verdict.reason, state, trajectory as never);
        }
      }
    } catch (error) {
      outcome = error instanceof IterationDivergedError ? 'divergent' : 'error';
      try { await emit({ kind: 'error', iteration: steps.length, error: (error as Error)?.message ?? String(error) }); } catch { /* consumer gone */ }
      throw error;
    } finally {
      if (persistent) await store.write(statsKey, updateStatistics(stats, steps.length, activeMs, Date.now() - started, outcome === 'done'));
      task.record({ callId: `${task.id}/${iterationId}`, parentCallId: frame.parentCallId ?? null, taskId: task.id,
        definitionId: `iterateOn:${siteId}`, name: `iterateOn ${siteId}`, outcome, detail: `${steps.length} steps`,
        events: [{ kind: 'iteration', iteration_id: iterationId, site_id: siteId, step: stepId,
          predicate: doneMeta?.definition.id ?? null, persistent_statistics: persistent, active_ms: activeMs,
          wall_ms: Date.now() - started, reviews, repeats: repeats.length }, ...traceEvents] });
    }
  }
}

/** Free form of the operator; `callable.iterateOn(initial, ...args)` is the same thing. */
export function iterateOn<T>(step: Step<T>, initial: T, ...fixed: unknown[]): Iteration<T> {
  return new Iteration(step, initial, fixed);
}
