/**
 * A small serial event loop for applications with state, events, and a view. It is ordinary
 * TypeScript: `reduce` and `view` are callbacks that may call natlang functions wherever useful.
 *
 * Guarantees: events are applied one at a time in arrival order; an event ID is applied at most
 * once; a new state is committed (`onCommit`) before it is published or viewed; a failed view can be
 * retried with `refresh()` without replaying the committed event; `cancel()` aborts the active step.
 */

export type AppEvent = { id: string; kind: string; [key: string]: unknown };
export type StepContext = { signal: AbortSignal; revision: number; stage: 'reduce' | 'view' };
export type Transition<S, V, E extends AppEvent> = { event: E | null; revision: number; state: S; view: V };
export type Commit<S, E extends AppEvent> = { event: E; revision: number; state: S };
export type Failure<E extends AppEvent> = { event: E | null; stage: 'reduce' | 'view' | 'commit'; revision: number; error: unknown };

export type EventLoopOptions<S, V, E extends AppEvent> = {
  initialState: S;
  initialRevision?: number;
  seenEventIds?: Iterable<string>;
  reduce(state: S, event: E, context: StepContext): S | Promise<S>;
  view(state: S, context: StepContext): V | Promise<V>;
  /** Persist a new state before it is published. A failure leaves the previous state in place. */
  onCommit?(commit: Commit<S, E>): void | Promise<void>;
  onTransition?(transition: Transition<S, V, E>): void;
  onFailure?(failure: Failure<E>): void;
  /** Wrap each step, for example with `runtime.run(...)` so natlang calls have a task. */
  step?<T>(fn: () => Promise<T>, context: StepContext): Promise<T>;
};

export class EventLoop<S, V, E extends AppEvent = AppEvent> {
  private readonly seen: Set<string>;
  private queue: Promise<unknown> = Promise.resolve();
  private active: AbortController | null = null;
  private started = false;
  private closed = false;
  private stateValue: S;
  private viewValue: V | null = null;
  private revisionValue: number;

  constructor(private readonly options: EventLoopOptions<S, V, E>) {
    if (options.initialRevision !== undefined && (!Number.isSafeInteger(options.initialRevision) || options.initialRevision < 0))
      throw new RangeError('initialRevision must be a nonnegative safe integer');
    this.stateValue = structuredClone(options.initialState);
    this.revisionValue = options.initialRevision ?? 0;
    this.seen = new Set(options.seenEventIds ?? []);
  }

  get state(): S { return structuredClone(this.stateValue); }
  get view(): V | null { return this.viewValue; }
  get revision(): number { return this.revisionValue; }
  get seenEventIds(): string[] { return [...this.seen]; }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('event loop is closed'));
    const execute = () => { if (this.closed) throw new Error('event loop is closed'); return task(); };
    const next = this.queue.then(execute, execute);
    this.queue = next.catch(() => undefined);
    return next;
  }

  private async stage<T>(stage: 'reduce' | 'view', fn: (context: StepContext) => T | Promise<T>): Promise<T> {
    const controller = new AbortController();
    this.active = controller;
    const context: StepContext = { signal: controller.signal, revision: this.revisionValue, stage };
    try {
      const run = async () => fn(context);
      return await (this.options.step ? this.options.step(run, context) : run());
    } finally { if (this.active === controller) this.active = null; }
  }

  private fail(event: E | null, stage: Failure<E>['stage'], error: unknown): never {
    this.options.onFailure?.({ event, stage, revision: this.revisionValue, error });
    throw error;
  }

  private async render(event: E | null): Promise<Transition<S, V, E>> {
    let view: V;
    try { view = await this.stage('view', context => this.options.view(this.state, context)); }
    catch (error) { return this.fail(event, 'view', error); }
    this.viewValue = view;
    const transition = { event, revision: this.revisionValue, state: this.state, view };
    this.options.onTransition?.(transition);
    return transition;
  }

  start(): Promise<Transition<S, V, E>> {
    return this.enqueue(async () => {
      if (this.started) throw new Error('event loop already started');
      this.started = true;
      return this.render(null);
    });
  }

  /** Apply one event. Resolves to null for an event ID that was already applied. */
  dispatch(event: E): Promise<Transition<S, V, E> | null> {
    if (!event || typeof event.id !== 'string' || !event.id || typeof event.kind !== 'string' || !event.kind)
      return Promise.reject(new TypeError('an event needs a string id and kind'));
    return this.enqueue(async () => {
      if (!this.started) throw new Error('event loop has not started');
      if (this.seen.has(event.id)) return null;
      let state: S;
      try { state = await this.stage('reduce', context => this.options.reduce(this.state, structuredClone(event), context)); }
      catch (error) { return this.fail(event, 'reduce', error); }
      try { await this.options.onCommit?.({ event, revision: this.revisionValue + 1, state: structuredClone(state) }); }
      catch (error) { return this.fail(event, 'commit', error); }
      this.stateValue = structuredClone(state);
      this.revisionValue++;
      this.seen.add(event.id);
      return this.render(event);
    });
  }

  /** Recompute the view of the current state (for example after a failed view). */
  refresh(): Promise<Transition<S, V, E>> {
    return this.enqueue(async () => {
      if (!this.started) throw new Error('event loop has not started');
      return this.render(null);
    });
  }

  cancel(): void { this.active?.abort(new Error('step cancelled')); }

  async consume(events: AsyncIterable<E>, each?: (transition: Transition<S, V, E>) => void | Promise<void>,
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
    this.active?.abort(new Error('event loop closed'));
    await this.queue;
  }
}

/** Push-driven async event stream for readline, jobs, file watchers, sockets, or UI events. */
export class EventQueue<E> implements AsyncIterable<E> {
  private values: E[] = [];
  private waiters: Array<{ resolve: (result: IteratorResult<E>) => void; reject: (error: unknown) => void }> = [];
  private ended = false;
  private failure: unknown = null;

  push(value: E): void {
    if (this.ended) throw new Error('event queue is closed');
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ done: false, value }); else this.values.push(value);
  }
  close(): void {
    if (this.ended) return;
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.resolve({ done: true, value: undefined });
  }
  fail(error: unknown): void {
    if (this.ended) return;
    this.failure = error; this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }
  [Symbol.asyncIterator](): AsyncIterator<E> {
    return { next: () => {
      if (this.values.length) return Promise.resolve({ done: false, value: this.values.shift()! });
      if (this.failure) return Promise.reject(this.failure);
      if (this.ended) return Promise.resolve({ done: true, value: undefined });
      return new Promise<IteratorResult<E>>((resolve, reject) => this.waiters.push({ resolve, reject }));
    } };
  }
}
