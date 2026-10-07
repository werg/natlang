/**
 * A small serial event loop for applications with state, events, and a view. It is ordinary
 * TypeScript: `reduce` and `view` are callbacks that may call natlang functions wherever useful.
 *
 * Guarantees: events are applied one at a time in arrival order; an event ID is applied at most
 * once; a new state is committed (`onCommit`) before it is published or viewed; a failed view can be
 * retried with `refresh()` without replaying the committed event; `cancel()` aborts the active step.
 *
 * Slow work can leave the queue: `context.after(work)` runs once the step's state is committed and
 * dispatches the event it returns (Elm's commands). Time is an input: `context.now` is fixed per step
 * and stored with the commit, and `wakeAt(state)` keeps one timer that dispatches a `wake` event, so
 * deadlines and reminders are state that survives a restart. `KeyedEventLoop` runs one loop per key
 * (a user, an order, a document): serial within a key, parallel across keys.
 */

export type AppEvent = { id: string; kind: string; [key: string]: unknown };
/** The event `wakeAt` dispatches; include it in a loop's event type when using `wakeAt`. */
export type WakeEvent = { id: string; kind: 'wake'; at: number };
export type StepContext = { signal: AbortSignal; revision: number; stage: 'reduce' | 'view';
  /** The time of this step (ms since the epoch, from `clock`), stored with its commit. */
  now: number;
  /**
   * Run `work` after this step's state is committed and dispatch the event it returns, if any. It is not awaited by
   * the step; `close()` aborts its signal. Available while reducing.
   */
  after(work: (signal: AbortSignal) => AppEvent | null | undefined | void | Promise<AppEvent | null | undefined | void>): void };
export type Transition<S, V, E extends AppEvent> = { event: E | null; revision: number; state: S; view: V };
export type Commit<S, E extends AppEvent> = { event: E; revision: number; state: S; now: number };
export type Failure<E extends AppEvent> = { event: E | null; stage: 'reduce' | 'view' | 'commit' | 'after'; revision: number; error: unknown };

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
  /**
   * When the loop should next receive a `wake` event, from its committed state (ms since the epoch), or null. Each time
   * fires once; return a later time to be woken again.
   */
  wakeAt?(state: S): number | null | undefined;
  /** The clock behind `context.now` and `wakeAt` (default `Date.now`). */
  clock?(): number;
};

/** Longest delay a timer accepts; a later wake-up re-arms when it fires. */
const MAX_TIMER_MS = 2 ** 31 - 1;

export class EventLoop<S, V, E extends AppEvent = AppEvent> {
  private readonly seen: Set<string>;
  private queue: Promise<unknown> = Promise.resolve();
  private active: AbortController | null = null;
  private started = false;
  private closed = false;
  private stateValue: S;
  private viewValue: V | null = null;
  private revisionValue: number;
  private readonly jobs = new Set<AbortController>();
  private wake?: { at: number; timer: ReturnType<typeof setTimeout> };
  /** The latest wake time already delivered: each time fires once, so a reducer that ignores it is not woken again. */
  private woke = -Infinity;

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

  private now(): number { return this.options.clock ? this.options.clock() : Date.now(); }

  private async stage<T>(stage: 'reduce' | 'view', fn: (context: StepContext) => T | Promise<T>,
    now = this.now(), followUps?: Parameters<StepContext['after']>[0][]): Promise<T> {
    const controller = new AbortController();
    this.active = controller;
    const context: StepContext = { signal: controller.signal, revision: this.revisionValue, stage, now,
      after: work => {
        if (!followUps) throw new Error('context.after is available while reducing an event');
        if (typeof work !== 'function') throw new TypeError('context.after needs a function');
        followUps.push(work);
      } };
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
      this.armWake();
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
      const now = this.now(), followUps: Parameters<StepContext['after']>[0][] = [];
      try { state = await this.stage('reduce', context => this.options.reduce(this.state, structuredClone(event), context), now, followUps); }
      catch (error) { return this.fail(event, 'reduce', error); }
      try { await this.options.onCommit?.({ event, revision: this.revisionValue + 1, state: structuredClone(state), now }); }
      catch (error) { return this.fail(event, 'commit', error); }
      this.stateValue = structuredClone(state);
      this.revisionValue++;
      this.seen.add(event.id);
      this.armWake();
      const revision = this.revisionValue;
      for (const work of followUps) this.runAfter(event, revision, work);
      return this.render(event);
    });
  }

  /** Follow-up work of a committed event: its resulting event is dispatched to this loop. */
  private runAfter(event: E, revision: number, work: Parameters<StepContext['after']>[0]): void {
    const controller = new AbortController();
    this.jobs.add(controller);
    void (async () => {
      try {
        const next = await work(controller.signal);
        if (next && !controller.signal.aborted && !this.closed) await this.dispatch(next as E);
      } catch (error) {
        if (!controller.signal.aborted) this.options.onFailure?.({ event, stage: 'after', revision, error });
      } finally { this.jobs.delete(controller); }
    })();
  }

  /** Keep one timer at the committed state's wake time; it dispatches a `wake` event when the time comes. */
  private armWake(): void {
    const at = this.options.wakeAt?.(this.stateValue);
    if (this.wake && this.wake.at === at) return;
    if (this.wake) { clearTimeout(this.wake.timer); this.wake = undefined; }
    if (typeof at !== 'number' || !Number.isFinite(at) || at <= this.woke || this.closed) return;
    const fire = () => {
      if (this.closed || this.wake?.at !== at) return;
      const remaining = at - this.now();
      if (remaining > 0) { this.wake = { at, timer: setTimeout(fire, Math.min(remaining, MAX_TIMER_MS)) }; return; }
      this.wake = undefined;
      this.woke = at;
      const wake: WakeEvent = { id: `wake:${this.revisionValue}:${at}`, kind: 'wake', at };
      this.dispatch(wake as unknown as E).catch(() => { /* reported through onFailure */ });
    };
    this.wake = { at, timer: setTimeout(fire, Math.min(Math.max(0, at - this.now()), MAX_TIMER_MS)) };
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
    for (const job of this.jobs) job.abort(new Error('event loop closed'));
    if (this.wake) { clearTimeout(this.wake.timer); this.wake = undefined; }
    await this.queue;
  }
}

export type KeyedEventLoopOptions<S, V, E extends AppEvent> =
  Omit<EventLoopOptions<S, V, E>, 'initialState' | 'initialRevision' | 'seenEventIds' | 'onCommit' | 'onTransition' | 'onFailure'> & {
  /** The key whose loop applies an event. */
  key(event: E): string;
  /** A new key's state. */
  initialState(key: string): S;
  /** A key's persisted state, revision and applied event IDs, if it has any (loaded once, before its first event). */
  restore?(key: string): { state: S; revision: number; seenEventIds?: Iterable<string> } | undefined |
    Promise<{ state: S; revision: number; seenEventIds?: Iterable<string> } | undefined>;
  onCommit?(key: string, commit: Commit<S, E>): void | Promise<void>;
  onTransition?(key: string, transition: Transition<S, V, E>): void;
  onFailure?(key: string, failure: Failure<E>): void;
};

/**
 * One `EventLoop` per key, created on its first event: events of one key apply in order, keys run in parallel, so one
 * user's slow step does not hold up another's.
 */
export class KeyedEventLoop<S, V, E extends AppEvent = AppEvent> {
  private readonly loops = new Map<string, Promise<EventLoop<S, V, E>>>();
  private closed = false;

  constructor(private readonly options: KeyedEventLoopOptions<S, V, E>) {}

  /** The started loop of a key. */
  loop(key: string): Promise<EventLoop<S, V, E>> {
    if (this.closed) return Promise.reject(new Error('event loop is closed'));
    let loop = this.loops.get(key);
    if (!loop) {
      const { key: _key, initialState, restore, onCommit, onTransition, onFailure, ...shared } = this.options;
      loop = (async () => {
        const saved = await restore?.(key);
        const created = new EventLoop<S, V, E>({ ...shared, initialState: saved ? saved.state : initialState(key),
          initialRevision: saved?.revision, seenEventIds: saved?.seenEventIds,
          onCommit: onCommit && (commit => onCommit(key, commit)), onTransition: onTransition && (transition => onTransition(key, transition)),
          onFailure: onFailure && (failure => onFailure(key, failure)) });
        await created.start();
        return created;
      })();
      this.loops.set(key, loop);
      loop.catch(() => this.loops.delete(key));
    }
    return loop;
  }

  /** Apply one event in its key's loop. */
  async dispatch(event: E): Promise<Transition<S, V, E> | null> {
    return (await this.loop(this.options.key(event))).dispatch(event);
  }

  /** The keys with a loop in this process. */
  keys(): string[] { return [...this.loops.keys()]; }

  async close(): Promise<void> {
    this.closed = true;
    await Promise.all([...this.loops.values()].map(loop => loop.then(item => item.close(), () => undefined)));
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
