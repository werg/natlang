import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { resolveApplicationInputs, type ApplicationInputs } from '../application-inputs.js';
import type { ModelTurn, RunOptions, RunResult } from '../contracts.js';
import { NativeNatlangHost, type NativeRunRequest } from '../native/host.js';

export type TerminalEvent = { id: string; kind: string; [key: string]: unknown };
export type TerminalSource = { reducer: string; view: string };
export type TerminalTransition<S, V, E extends TerminalEvent = TerminalEvent> = {
  event: E | null; revision: number; state: S; view: V;
  reducerRun: RunResult | null; viewRun: RunResult;
};
export type TerminalFailure<E extends TerminalEvent = TerminalEvent> = {
  event: E | null; stage: 'reduce' | 'view'; revision: number;
  detail: string; run: RunResult | null;
};
export type TerminalCommit<S, E extends TerminalEvent = TerminalEvent> = {
  event: E; revision: number; state: S; reducerRun: RunResult;
};
export type TerminalRunner = { run(request: NativeRunRequest): Promise<RunResult> };

export type TerminalApplicationOptions<S, V, E extends TerminalEvent = TerminalEvent> = {
  runner: TerminalRunner;
  source: TerminalSource;
  /** Inputs supplied to both reducer and view runs; factories are evaluated per run. */
  inputs?: ApplicationInputs;
  /** Inputs supplied only to the semantic reducer; factories are evaluated per run. */
  reducerInputs?: ApplicationInputs;
  /** Inputs supplied only to the view; factories are evaluated per run. */
  viewInputs?: ApplicationInputs;
  initialState: S;
  initialRevision?: number;
  seenEventIds?: Iterable<string>;
  modelTurn?: (request: Parameters<NonNullable<NativeRunRequest['modelTurn']>>[0]) => Promise<ModelTurn> | ModelTurn;
  runOptions?: Omit<RunOptions, 'seed' | 'run_id'>;
  validationFeedback?: NativeRunRequest['validationFeedback'];
  seedRoot?: number;
  traceDirectory?: string;
  onCommit?: (commit: TerminalCommit<S, E>) => void | Promise<void>;
  onTransition?: (transition: TerminalTransition<S, V, E>) => void;
  onFailure?: (failure: TerminalFailure<E>) => void;
};

function eventSeed(root: number, revision: number, id: string): number {
  let value = root >>> 0;
  for (const char of `${revision}:${id}`) {
    value ^= char.charCodeAt(0);
    value = Math.imul(value, 16777619) >>> 0;
  }
  return value;
}

function safeTraceName(id: string): string {
  return id.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 120) || 'event';
}

/** A serial event reducer plus independently retryable terminal view. */
export class TerminalNatlangApplication<S, V, E extends TerminalEvent = TerminalEvent> {
  private readonly options: TerminalApplicationOptions<S, V, E>;
  private readonly seen: Set<string>;
  private queue: Promise<unknown> = Promise.resolve();
  private active: AbortController | null = null;
  private started = false;
  private closed = false;
  private stateValue: S;
  private viewValue: V | null = null;
  private revisionValue: number;
  private runSequence = 0;

  constructor(options: TerminalApplicationOptions<S, V, E>) {
    if (!options.source.reducer || !options.source.view) throw new Error('terminal application needs reducer and view sources');
    if (options.seedRoot !== undefined && !Number.isSafeInteger(options.seedRoot))
      throw new Error('seedRoot must be a safe integer');
    if (options.initialRevision !== undefined && (!Number.isSafeInteger(options.initialRevision) || options.initialRevision < 0))
      throw new Error('initialRevision must be a nonnegative safe integer');
    this.options = options;
    this.stateValue = structuredClone(options.initialState);
    this.revisionValue = options.initialRevision ?? 0;
    this.seen = new Set(options.seenEventIds ?? []);
  }

  get state(): S { return structuredClone(this.stateValue); }
  get view(): V | null { return this.viewValue === null ? null : structuredClone(this.viewValue); }
  get revision(): number { return this.revisionValue; }
  get seenEventIds(): string[] { return [...this.seen]; }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('terminal application is closed'));
    const execute = () => {
      if (this.closed) throw new Error('terminal application is closed');
      return task();
    };
    const next = this.queue.then(execute, execute);
    this.queue = next.catch(() => undefined);
    return next;
  }

  private async run(path: string, inputs: Record<string, unknown>, revision: number,
    eventId: string, stage: 'reduce' | 'view'): Promise<RunResult> {
    const controller = new AbortController();
    this.active = controller;
    const sequence = ++this.runSequence;
    let tracePath: string | undefined;
    if (this.options.traceDirectory) {
      mkdirSync(this.options.traceDirectory, { recursive: true });
      tracePath = join(this.options.traceDirectory,
        `${String(revision).padStart(8, '0')}-${safeTraceName(eventId)}-${stage}-${sequence}.jsonl`);
    }
    try {
      const shared = resolveApplicationInputs(this.options.inputs);
      const specific = stage === 'reduce' ? this.options.reducerInputs : this.options.viewInputs;
      const stageInputs = resolveApplicationInputs(specific);
      return await this.options.runner.run({ source: { kind: 'file', path },
        inputs: { ...shared, ...stageInputs, ...inputs },
        modelTurn: this.options.modelTurn,
        validationFeedback: this.options.validationFeedback ?? 'local',
        signal: controller.signal, tracePath,
        options: { ...this.options.runOptions,
          run_id: `terminal:${revision}:${eventId}:${stage}:${sequence}`,
          ...(this.options.seedRoot === undefined ? {} : {
            seed: { mode: 'derived', root: eventSeed(this.options.seedRoot, revision, eventId) },
          }) },
      });
    } finally { if (this.active === controller) this.active = null; }
  }

  private fail(event: E | null, stage: 'reduce' | 'view', run: RunResult | null,
    detail: string): never {
    this.options.onFailure?.({ event, stage, revision: this.revisionValue, detail, run });
    throw new Error(`${stage} failed: ${detail}`);
  }

  private async render(event: E | null, reducerRun: RunResult | null): Promise<TerminalTransition<S, V, E>> {
    let viewRun: RunResult;
    try {
      viewRun = await this.run(this.options.source.view, { state: this.state }, this.revisionValue,
        event?.id ?? 'initial-view', 'view');
    } catch (error) { return this.fail(event, 'view', null, String(error)); }
    if (viewRun.outcome.kind !== 'done') return this.fail(event, 'view', viewRun, viewRun.outcome.detail);
    this.viewValue = structuredClone(viewRun.value as V);
    const transition = { event, revision: this.revisionValue, state: this.state,
      view: this.view as V, reducerRun, viewRun };
    try { this.options.onTransition?.(transition); }
    catch (error) { return this.fail(event, 'view', viewRun, String(error)); }
    return transition;
  }

  start(): Promise<TerminalTransition<S, V, E>> {
    return this.enqueue(async () => {
      if (this.started) throw new Error('terminal application already started');
      this.started = true;
      return this.render(null, null);
    });
  }

  dispatch(event: E): Promise<TerminalTransition<S, V, E> | null> {
    if (!event || typeof event.id !== 'string' || !event.id || typeof event.kind !== 'string' || !event.kind)
      return Promise.reject(new Error('terminal event needs string id and kind'));
    return this.enqueue(async () => {
      if (!this.started) throw new Error('terminal application has not started');
      if (this.seen.has(event.id)) return null;
      let reducerRun: RunResult;
      try {
        reducerRun = await this.run(this.options.source.reducer,
          { state: this.state, event: structuredClone(event) }, this.revisionValue, event.id, 'reduce');
      } catch (error) { return this.fail(event, 'reduce', null, String(error)); }
      if (reducerRun.outcome.kind !== 'done') return this.fail(event, 'reduce', reducerRun, reducerRun.outcome.detail);
      const state = structuredClone(reducerRun.value as S);
      try {
        await this.options.onCommit?.({ event, revision: this.revisionValue + 1,
          state: structuredClone(state), reducerRun });
      } catch (error) { return this.fail(event, 'reduce', reducerRun, `commit failed: ${String(error)}`); }
      this.stateValue = state;
      this.revisionValue++;
      this.seen.add(event.id);
      return this.render(event, reducerRun);
    });
  }

  refresh(): Promise<TerminalTransition<S, V, E>> {
    return this.enqueue(async () => {
      if (!this.started) throw new Error('terminal application has not started');
      return this.render(null, null);
    });
  }

  cancel(): void { this.active?.abort(); }
  async consume(events: AsyncIterable<E>,
    each?: (transition: TerminalTransition<S, V, E>) => void | Promise<void>,
    failed?: (error: unknown, event: E) => void | Promise<void>): Promise<void> {
    for await (const event of events) {
      try {
        const transition = await this.dispatch(event);
        if (transition) await each?.(transition);
      } catch (error) {
        if (!failed) throw error;
        await failed(error, event);
      }
    }
  }
  async close(): Promise<void> {
    this.closed = true;
    this.active?.abort();
    await this.queue;
  }
}

/** Convenience owner for applications that do not already manage a native host. */
export function createTerminalHost(host: object, mode: 'fresh' | 'retained' = 'retained'): NativeNatlangHost {
  return new NativeNatlangHost({ host, mode });
}
