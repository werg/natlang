/** Host event ingress for selected callable programs. Arrival is observed data;
 * callbacks run through the caller's selected executor, never a polling model. */
import { canonical, fingerprint } from '../adaptation/identity.js';

export type SelectedEvent<T> = Readonly<{ source: string; id: string; position: number; value: T }>;
export type EventReceipt = Readonly<{ source: string; id?: string; position?: number;
  disposition: 'buffered' | 'delivered' | 'duplicate' | 'overflow' | 'closed' | 'failed' | 'cancelled' }>;
export type EventQueueSnapshot<T> = {
  schema: 'natlang-event-queue/1'; source: string; capacity: number;
  historyCapacity: number;
  nextPosition: number; state: 'open' | 'closed';
  buffered: SelectedEvent<T>[]; seen: [string, { position: number; digest: string }][];
};

function eventData(value: unknown, active = new Set<object>()): unknown {
  if (value === null || ['string', 'boolean'].includes(typeof value)) return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (!value || typeof value !== 'object' || active.has(value)) throw new TypeError('Event payloads require acyclic finite JSON data');
  const array = Array.isArray(value), prototype = Object.getPrototypeOf(value);
  if (!array && prototype !== null && prototype !== Object.prototype) throw new TypeError('Event payloads require ordinary data records');
  active.add(value);
  const descriptors = Object.getOwnPropertyDescriptors(value), result = array ? [] : Object.create(null);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (array && key === 'length') continue;
    const descriptor = Reflect.get(descriptors, key);
    if (typeof key !== 'string' || !('value' in descriptor) || !descriptor.enumerable)
      throw new TypeError('Event payloads cannot contain accessors, symbols or hidden properties');
    Object.defineProperty(result, key, { value: eventData(descriptor.value, active), enumerable: true, writable: true, configurable: true });
  }
  if (array && (result.length !== descriptors.length!.value || Object.keys(result).length !== result.length))
    throw new TypeError('Event payload arrays must be dense JSON arrays');
  active.delete(value); return result;
}

export class SelectedEventQueue<T> implements AsyncIterableIterator<SelectedEvent<T>> {
  private readonly buffered: SelectedEvent<T>[] = [];
  private readonly seen = new Map<string, { position: number; digest: string }>();
  private readonly space: { resolve(): void; reject(error: unknown): void }[] = [];
  private waiting?: { resolve(value: IteratorResult<SelectedEvent<T>>): void; reject(error: unknown): void };
  private state: 'open' | 'closed' | 'failed' | 'cancelled' = 'open';
  private failure?: unknown;
  private position = 0;
  constructor(readonly source: string, readonly capacity: number,
    private readonly receipt: (receipt: EventReceipt) => void = () => {}, readonly historyCapacity = 65536) {
    if (typeof source !== 'string' || !source || !Number.isSafeInteger(capacity) || capacity < 1 ||
        !Number.isSafeInteger(historyCapacity) || historyCapacity < capacity)
      throw new TypeError('Event ingress needs a source identity and positive bounded capacity');
  }
  [Symbol.asyncIterator](): AsyncIterableIterator<SelectedEvent<T>> { return this; }
  get pending(): number { return this.buffered.length; }
  get status(): string { return this.state; }
  private open(): void {
    if (this.state !== 'open') throw this.failure ?? new Error(`Event source is ${this.state}`);
  }
  /** A pausable source awaits admission. Concurrent producers remain bounded. */
  async push(id: string, value: T): Promise<EventReceipt> {
    if (typeof id !== 'string' || !id) throw new TypeError('Event delivery requires a stable identity');
    const text = canonical(eventData(value)), digest = fingerprint(text, 'natlang.event/1');
    const known = this.seen.get(id);
    if (known) {
      if (known.digest !== digest) throw new Error('Event identity was reused with another payload');
      const receipt = { source: this.source, id, position: known.position, disposition: 'duplicate' as const };
      this.receipt(receipt); return receipt;
    }
    this.open();
    while (this.buffered.length >= this.capacity && !this.seen.has(id)) {
      this.open();
      if (this.space.length >= this.capacity) {
        this.receipt({ source: this.source, id, disposition: 'overflow' });
        throw new RangeError('Event ingress waiting producers exceed its bound');
      }
      await new Promise<void>((resolve, reject) => this.space.push({ resolve, reject }));
    }
    const receipt = this.post(id, JSON.parse(text) as T);
    if (this.buffered.length < this.capacity) this.space.shift()?.resolve();
    return receipt;
  }
  /** A source that cannot pause gets an explicit overflow disposition. */
  post(id: string, value: T): EventReceipt {
    if (typeof id !== 'string' || !id) throw new TypeError('Event delivery requires a stable identity');
    const text = canonical(eventData(value)), digest = fingerprint(text, 'natlang.event/1');
    const known = this.seen.get(id);
    if (known) {
      if (known.digest !== digest) throw new Error('Event identity was reused with another payload');
      const receipt = { source: this.source, id, position: known.position, disposition: 'duplicate' as const };
      this.receipt(receipt); return receipt;
    }
    this.open();
    if (this.seen.size >= this.historyCapacity) {
      this.receipt({ source: this.source, id, disposition: 'overflow' });
      throw new RangeError('Event delivery history is full; checkpoint and rotate the declared source epoch');
    }
    if (this.buffered.length >= this.capacity) {
      this.receipt({ source: this.source, id, disposition: 'overflow' });
      throw new RangeError('Event queue overflow; pause the producer or increase its declared capacity');
    }
    const event = Object.freeze({ source: this.source, id, position: this.position++, value: JSON.parse(text) as T });
    this.seen.set(id, { position: event.position, digest });
    const receipt = { source: this.source, id, position: event.position, disposition: 'buffered' as const };
    this.receipt(receipt);
    if (this.waiting) {
      const waiting = this.waiting; this.waiting = undefined;
      this.receipt({ ...receipt, disposition: 'delivered' });
      waiting.resolve({ value: event, done: false });
    } else this.buffered.push(event);
    return receipt;
  }
  next(): Promise<IteratorResult<SelectedEvent<T>>> {
    if (this.buffered.length) {
      const value = this.buffered.shift()!;
      this.space.shift()?.resolve();
      this.receipt({ source: this.source, id: value.id, position: value.position, disposition: 'delivered' });
      return Promise.resolve({ value, done: false });
    }
    if (this.state === 'failed' || this.state === 'cancelled') return Promise.reject(this.failure);
    if (this.state === 'closed') return Promise.resolve({ value: undefined, done: true });
    if (this.waiting) return Promise.reject(new Error('Event source has one owning consumer'));
    return new Promise((resolve, reject) => { this.waiting = { resolve, reject }; });
  }
  close(): void {
    if (this.state !== 'open') return;
    this.state = 'closed'; this.receipt({ source: this.source, disposition: 'closed' });
    for (const writer of this.space.splice(0)) writer.reject(new Error('Event source closed before admission'));
    this.waiting?.resolve({ value: undefined, done: true }); this.waiting = undefined;
  }
  fail(error: unknown): void { this.stop('failed', error); }
  cancel(error: unknown = new Error('Event source cancelled')): void { this.stop('cancelled', error); }
  private stop(state: 'failed' | 'cancelled', error: unknown): void {
    if (this.state === 'failed' || this.state === 'cancelled' || (this.state === 'closed' && state !== 'cancelled')) return;
    this.state = state; this.failure = error;
    this.receipt({ source: this.source, disposition: state });
    for (const writer of this.space.splice(0)) writer.reject(error);
    this.waiting?.reject(error); this.waiting = undefined;
    // Cancellation does not admit work that was only buffered. Failure retains
    // accepted items for ordered delivery before reporting the source failure.
    if (state === 'cancelled') for (const event of this.buffered.splice(0))
      this.receipt({ source: this.source, id: event.id, position: event.position, disposition: 'cancelled' });
  }
  return(): Promise<IteratorResult<SelectedEvent<T>>> {
    this.cancel(); return Promise.resolve({ value: undefined, done: true });
  }
  checkpoint(): EventQueueSnapshot<T> {
    if (this.waiting || this.space.length || !['open', 'closed'].includes(this.state))
      throw new Error('Checkpoint event ingress at a settled admission boundary');
    return structuredClone({ schema: 'natlang-event-queue/1', source: this.source, capacity: this.capacity, historyCapacity: this.historyCapacity,
      nextPosition: this.position, state: this.state as 'open' | 'closed', buffered: this.buffered, seen: [...this.seen] });
  }
  static restore<T>(snapshot: EventQueueSnapshot<T>, receipt?: (receipt: EventReceipt) => void): SelectedEventQueue<T> {
    if (snapshot?.schema !== 'natlang-event-queue/1' || !['open', 'closed'].includes(snapshot.state) ||
      !Number.isSafeInteger(snapshot.nextPosition) || snapshot.nextPosition < 0 ||
      !Array.isArray(snapshot.buffered) || !Array.isArray(snapshot.seen))
      throw new TypeError('Invalid selected event checkpoint');
    const queue = new SelectedEventQueue<T>(snapshot.source, snapshot.capacity, receipt, snapshot.historyCapacity);
    if (snapshot.seen.length > queue.historyCapacity) throw new RangeError('Event checkpoint exceeds history capacity');
    if (snapshot.buffered.length > snapshot.capacity) throw new RangeError('Event checkpoint exceeds queue capacity');
    const positions = new Set<number>();
    for (const [id, record] of snapshot.seen) {
      if (typeof id !== 'string' || !id || queue.seen.has(id) || !Number.isSafeInteger(record.position) || record.position < 0 ||
        record.position >= snapshot.nextPosition || positions.has(record.position) || !/^[0-9a-f]{64}$/.test(record.digest))
        throw new TypeError('Invalid event delivery ledger');
      positions.add(record.position);
      queue.seen.set(id, { ...record });
    }
    if (positions.size !== snapshot.nextPosition) throw new TypeError('Event checkpoint is missing delivery history');
    let prior = -1;
    for (const event of snapshot.buffered) {
      const known = queue.seen.get(event.id);
      if (event.source !== snapshot.source || event.position <= prior || known?.position !== event.position ||
        known.digest !== fingerprint(canonical(eventData(event.value)), 'natlang.event/1'))
        throw new TypeError('Event checkpoint payload or order disagrees with delivery ledger');
      prior = event.position; queue.buffered.push(structuredClone(event));
    }
    queue.position = snapshot.nextPosition; queue.state = snapshot.state; return queue;
  }
}

/** Each selected callback completes before the next event is consumed. */
export async function foldSelectedEvents<T, S>(source: AsyncIterable<SelectedEvent<T>>, initial: S,
  step: (state: S, value: T, event: SelectedEvent<T>) => Promise<S> | S): Promise<S> {
  let state = initial;
  for await (const event of source) state = await step(state, event.value, event);
  return state;
}

/** Bounded independent execution with input-order results and source identity.
 * Completed later items occupy the same bounded window until earlier work ends. */
export async function* mapSelectedEvents<T, R>(source: AsyncIterable<SelectedEvent<T>>,
  select: (value: T, event: SelectedEvent<T>, signal: AbortSignal) => Promise<R> | R, concurrency: number):
  AsyncGenerator<SelectedEvent<R>> {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1)
    throw new RangeError('Selected Map concurrency must be a positive integer');
  const iterator = source[Symbol.asyncIterator]();
  type Outcome = { value: SelectedEvent<R> } | { error: unknown };
  type Input = { kind: 'input'; next: IteratorResult<SelectedEvent<T>> } | { kind: 'input-error'; error: unknown };
  const pending: Promise<Outcome>[] = [];
  const abort = new AbortController();
  let closed = false, reading: Promise<Input> | null = null;
  try {
    while (!closed || pending.length) {
      if (!closed && !reading && pending.length < concurrency)
        reading = Promise.resolve(iterator.next()).then(next => ({ kind: 'input', next }),
          error => ({ kind: 'input-error', error }));
      const candidates: Promise<Input | { kind: 'output'; result: Outcome }>[] = [];
      if (reading) candidates.push(reading);
      if (pending.length) candidates.push(pending[0]!.then(result => ({ kind: 'output', result })));
      const ready = await Promise.race(candidates);
      if (ready.kind === 'input-error') throw ready.error;
      if (ready.kind === 'input') {
        reading = null;
        if (ready.next.done) { closed = true; continue; }
        const event = ready.next.value;
        pending.push(Promise.resolve().then(() => select(event.value, event, abort.signal)).then(
          value => ({ value: Object.freeze({ ...event, value }) }), error => ({ error })));
      } else {
        pending.shift();
        if ('error' in ready.result) throw ready.result.error;
        yield ready.result.value;
      }
    }
  } finally {
    abort.abort(new Error('Selected Map consumer closed'));
    if (!closed) await iterator.return?.();
    // Every in-flight promise has a rejection handler. Cancellation is a request
    // to its adapter, not a claim that an already dispatched write was undone.
    void Promise.all(pending);
  }
}
