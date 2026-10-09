/**
 * `iterateOn`: the sanctioned open-ended iteration operator.
 *
 * The step is called sequentially with the current state and fixed arguments; its result is the
 * next state. The stopping predicate is checked on the initial state and after every step. Progress
 * reviews are scheduled from per-site statistics and never stop a run by themselves: only a
 * `divergent` verdict does.
 */
import { FolderSnapshot } from '../native/scoped-fs.js';
import { hexDigest } from '../native/hash.js';
import { isLive, liveId } from '../native/values.js';
import { currentFrame, runInFrame } from './context.js';
import { callableMeta } from './callable.js';
import { invokeDefinition, type CallableDefinition } from './kernel.js';
import { resolveFrame } from './runtime.js';
import { graphNode, invocationNodeId, traceFor, valueInputs } from '../native/graph.js';

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
  /** `iteration-unbounded` when a TypeScript predicate (or review-off loop) has neither measure nor step limit. */
  readonly code?: 'iteration-unbounded';
  constructor(message: string, lastState: unknown, trajectory: IterationTrajectory<unknown>, code?: 'iteration-unbounded') {
    super(code ? `${code}: ${message}` : message, lastState, trajectory); this.name = 'IterationLimitError';
    if (code) this.code = code;
  }
}

function stableHash(value: unknown): string {
  const seen = new WeakSet<object>();
  const canonical = (item: unknown): unknown => {
    if (item === null || typeof item !== 'object') return typeof item === 'function' ? `live:${liveId(item)}` : item;
    if (item instanceof FolderSnapshot) return `source:${item.digest}`;
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

export const DEFAULT_JUDGE_INSTRUCTIONS = `An iterative process has been running for a while. Decide whether it is still making meaningful progress.
Inspect the trajectory: use trajectory.summary(), trajectory.steps(), trajectory.repeats(), and trajectory.states(start, end) to page through states.
Distinguish real improvement from repetition, oscillation between the same few states, unproductive churn, and a goal that looks impossible.
An unusually long run that is still improving should continue.
Return { verdict: "continue", reason } if further steps are likely to help, or { verdict: "divergent", reason } if the process is stuck or cannot succeed. Give a concrete reason.`;

/**
 * System prompt addition for a natural-language stopping predicate. Such a loop has no hard bound, so its predicate is
 * told what it is and biased against holding out for a state the loop will never reach; a loop that is stuck is the
 * progress judge's to stop.
 */
export function predicatePrompt(completedSteps: number, unchangedSteps: number): string {
  return [
    'This call is the stopping condition of an iterative loop (iterateOn). Answer true to stop the loop with the current ' +
      'state as its result, or false to run another step.',
    `The loop has completed ${completedSteps} step${completedSteps === 1 ? '' : 's'}.` +
      (completedSteps === 0 ? ' This is the initial state, before any step.' :
        unchangedSteps > 0 ? ` The state has not changed over the last ${unchangedSteps} step${unchangedSteps === 1 ? '' : 's'}.` :
          ' The last step changed the state.'),
    'Judge whether the criterion is met in substance by the current state. Accept a state that reasonably meets it; do not ' +
      'hold out for perfection or for details the criterion does not ask for, since every further step costs a model call ' +
      'and the loop ends only when you accept.',
    'If the state has stopped changing, further steps are unlikely to change your answer: if it meets the criterion in ' +
      'substance, answer true. Do not answer true for a state that fails the criterion; a loop that cannot succeed is ' +
      'stopped by its progress review, not by this answer.',
  ].join('\n');
}

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
  private judge?: ProgressJudgeFunction | 'off';
  private siteId?: string;
  private compilerSite?: string;
  private limit: { maxSteps?: number; deadlineMs?: number } = {};
  private measure?: (state: T) => number;
  private started = false;
  private frame?: import('./context.js').Frame;
  readonly [Symbol.toStringTag] = 'Iteration';

  constructor(private readonly step: Step<T>, private readonly initial: T, private readonly fixed: unknown[]) {
    if (typeof step !== 'function') throw new TypeError('iterateOn requires a step function');
  }

  onStep(observer: (event: IterationEvent<T>) => void | Promise<void>): this { this.observers.push(observer); return this; }
  checkProgress(judge: ProgressJudgeFunction | 'off'): this { this.judge = judge; return this; }
  /** A well-founded workflow measure, independent of semantic reviews. */
  withMeasure(remaining: (state: T) => number): this {
    if (typeof remaining !== 'function') throw new TypeError('withMeasure requires a function');
    this.measure = remaining; return this;
  }
  withSiteId(id: string): this {
    if (!id.trim()) throw new TypeError('withSiteId requires a non-empty identifier');
    this.siteId = id; return this;
  }
  withLimit(limit: { maxSteps?: number; deadlineMs?: number }): this { for (const [name, value] of Object.entries(limit)) { if (!Number.isFinite(value) || value < 0 || (name === 'maxSteps' && !Number.isSafeInteger(value))) throw new RangeError(`invalid iteration limit: ${name}`); } this.limit = { ...limit }; return this; }
  /** Run in a fixed task frame (used when an interpreter session creates the iteration). */
  inFrame(frame: import('./context.js').Frame | undefined): this { this.frame ??= frame; return this; }
  /** Set by compiled code with the call site's stable identity. */
  withCompilerSite(id: string): this { this.compilerSite ??= id; return this; }

  until(done: Done<T>): Promise<T> {
    const frame = this.frame ?? currentFrame() ?? resolveFrame();
    // An eval can start an iteration without awaiting `.until()`. Track it like an
    // async Natlang call so its rejection is observed and drained with its caller,
    // rather than surfacing later as an unhandled worker rejection. Returning the
    // original promise preserves normal `await` and `Promise.all` error semantics.
    return frame.task.track(this.run(done, async () => {}, currentFrame()?.signal), frame.parentCallId);
  }

  /** What a printed iteration is: nothing runs until `.until(done)` is awaited. */
  toString(): string {
    return this.started ? 'iterateOn(…), running or run by .until' :
      'iterateOn(…) that has not run: it runs when you await .until(done), as in ' +
      'await iterateOn(step, initial).until(state => done(state)), and gives the final state';
  }

  streamUntil(done: Done<T>): AsyncIterable<IterationEvent<T>> & { readonly __natlangIterationStream: true } {
    const signal = currentFrame()?.signal;
    const queue: IterationEvent<T>[] = [];
    let wake: (() => void) | undefined, finished = false, failure: unknown, cancelled = false;
    let running: Promise<T> | undefined;
    const start = () => {
      running ??= this.run(done, async event => {
        queue.push(event); wake?.();
        if (cancelled) throw new Error('iteration stream consumer stopped');
      }, signal).then(value => { finished = true; wake?.(); return value; },
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

  private async run(done: Done<T>, emitStream: (event: IterationEvent<T>) => Promise<void>, signal?: AbortSignal): Promise<T> {
    if (this.started) throw new Error('an Iteration can be started only once; call iterateOn again for another run');
    this.started = true;
    // The iteration stops with the eval or call that started it; its steps' calls derive their signal from it.
    const base = this.frame ?? currentFrame() ?? resolveFrame();
    const frame = signal ? { ...base, signal } : base;
    const task = frame.task;
    const iterationId = `iter-${Math.random().toString(36).slice(2, 10)}`;
    const stepMeta = callableMeta(this.step), doneMeta = callableMeta(done);
    const stepId = stepMeta?.definition.id ?? (this.step.name || 'step');
    const siteId = this.siteId ?? this.compilerSite ?? `unsited:${stepId}`;
    const persistent = !!(this.siteId ?? this.compilerSite);
    const store = task.runtime.options.statistics ?? task.runtime.callStore()?.iterationStatistics?.() ?? defaultStatistics;
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
      // Each step is a node of the calling invocation's graph, with its state's soft values as inputs.
      if (full.kind !== 'error') graphNode(traceFor(frame.parentCallId), 'iteration_step', { iteration_id: iterationId, site_id: siteId,
        phase: full.kind, step: full.iteration, state_hash: full.state === undefined ? null : stableHash(full.state),
        predicate: full.kind === 'done' ? true : null, review: full.review ?? null },
        [...(frame.parentCallId ? [{ node: invocationNodeId(frame.parentCallId), port: 'caller' }] : []),
          ...(full.state === undefined ? [] : valueInputs(full.state, 'state'))]);
    };
    const traceEvents: Record<string, unknown>[] = [];
    let state = this.initial;
    let outcome = 'done';
    const measure = (value: T): number | undefined => {
      if (!this.measure) return undefined;
      try {
        const remaining = this.measure(value);
        if (!Number.isSafeInteger(remaining) || remaining < 0)
          throw new RangeError('iteration measure must be a nonnegative safe integer');
        return remaining;
      } catch (error) {
        throw new IterationStepError(`iteration measure failed: ${(error as Error)?.message ?? error}`, state, trajectory as never, error);
      }
    };
    let remaining: number | undefined;
    // A natural-language predicate runs under the stopping-condition prompt (S0 §8): how many steps ran, and whether
    // the state has stopped changing.
    const unchanged = (): number => {
      let count = 0;
      for (let index = steps.length - 1; index >= 0; index--) {
        const previous = index === 0 ? stableHash(this.initial) : steps[index - 1]!.stateHash;
        if (steps[index]!.stateHash !== previous) break;
        count++;
      }
      return count;
    };
    const check = async (): Promise<boolean> => {
      try {
        if (doneMeta) {
          const base = doneMeta.bound ?? frame;
          return await doneMeta.invoke([state], { ...base, systemAddendum: predicatePrompt(steps.length, unchanged()) }) === true;
        }
        return await runInFrame(frame, () => done(state)) === true;
      }
      catch (error) { throw new IterationStepError(`stopping predicate failed: ${(error as Error)?.message ?? error}`, state, trajectory as never, error); }
    };
    try {
      await emit({ kind: 'initial', iteration: 0, state });
      remaining = measure(state);
      if (await check()) { await emit({ kind: 'done', iteration: 0, state }); return state; }
      // A TypeScript predicate can loop forever, so it needs a hard bound; so does a loop whose progress judge is off.
      // A natural-language predicate under its stopping-condition prompt, watched by the progress judge, does not.
      if (!this.measure && this.limit.maxSteps === undefined && (!doneMeta || judge === 'off'))
        throw new IterationLimitError(doneMeta ?
          'an iteration with progress review off requires withLimit({ maxSteps }) or withMeasure(...)' :
          'an iteration with a TypeScript stopping predicate requires withLimit({ maxSteps }) or withMeasure(...); ' +
          'a natural-language predicate (until(nl`…`)) needs neither', state, trajectory as never, 'iteration-unbounded');
      while (true) {
        task.checkOpen();
        frame.signal?.throwIfAborted();
        if (remaining === 0)
          throw new IterationLimitError('iteration exhausted its remaining work measure', state, trajectory as never);
        if (this.limit.maxSteps !== undefined && steps.length >= this.limit.maxSteps)
          throw new IterationLimitError(`iteration reached its limit of ${this.limit.maxSteps} steps`, state, trajectory as never);
        if (this.limit.deadlineMs !== undefined && Date.now() - started >= this.limit.deadlineMs)
          throw new IterationLimitError(`iteration reached its deadline of ${this.limit.deadlineMs} ms`, state, trajectory as never);
        const before = Date.now();
        let next: T;
        try { next = await runInFrame(frame, () => this.step(state, ...this.fixed)); }
        catch (error) { throw new IterationStepError(`step ${steps.length + 1} failed: ${(error as Error)?.message ?? error}`, state, trajectory as never, error); }
        const nextRemaining = measure(next);
        if (remaining !== undefined && nextRemaining! >= remaining)
          throw new IterationDivergedError('remaining work measure did not decrease', state, trajectory as never);
        remaining = nextRemaining;
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
        if (judge !== 'off' && reviewDue(stats, iteration, activeMs, lastReviewAt)) {
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
