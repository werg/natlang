/**
 * One scheduler per model backend (plans/BATCHED_EXECUTION.md §3.1). Every request for a backend waits here, in one
 * queue, instead of in a per-call-site limit. The scheduler decides when a request is sent and in what company:
 *
 * - It sends up to `maxConcurrent` requests at once. With a server that batches continuously (llama.cpp
 *   `--parallel N`, vLLM) that is the batch.
 * - Requests that become ready together are released together, in one coalescing window, and share a batch id.
 * - A turn of a call that is already running is served before the first turn of a new call, so started work finishes,
 *   frees its slot and KV, and a program waiting on one result stays fast. The scheduler recognises a running call
 *   by the invocation id the request carries (`ModelTurnRequest.invocation_id`): no turn counter is threaded through
 *   the runtime.
 * - Within a priority class, requests with the same prompt prefix (same system message) are sent adjacent, so the
 *   server's prefix cache or a slot with a matching prompt is reused.
 * - A queued request is cancelled by its AbortSignal, or all of an invocation's requests by `cancel`.
 *
 * `requestLimit` (chat-completion.ts) stays as the plain FIFO limit. A scheduler with `priority: false`, `adjacency:
 * false` and `coalesceMs: 0` admits requests exactly as a request limit of the same size does (tested).
 */
import type { ChatRequestMeta, ChatTransport, RequestLimit } from './chat-completion.js';

type Json = Record<string, unknown>;

/** How a backend turns concurrent requests into forward batches. */
export type BatchingMode =
  /** The server batches concurrent requests itself (llama.cpp `--parallel N`, vLLM, the Neuralese server). */
  | 'server-continuous'
  /** One request can carry many prompts (a scoring endpoint); the scheduler packs. Turns behave as continuous. */
  | 'explicit-batch'
  /** Remote APIs where batching buys nothing: the scheduler only limits concurrency. */
  | 'serial';
export type BackendCapabilities = { batching: { mode: BatchingMode; maxConcurrent: number } };

/** What the scheduler records about one request, delivered to the request's `onScheduled` hook when it is sent. */
export type ScheduleInfo = {
  batch_id: string;
  /** Requests released together with this one. */
  batch_size: number;
  /** Requests in flight on the backend when this one was sent, itself included. */
  in_flight: number;
  /** Time spent waiting in the scheduler's queue. */
  queue_wait_ms: number;
  priority: 'running' | 'new';
};

export type SchedulerOptions = {
  mode?: BatchingMode;
  /** Requests in flight at once. Infinity means no limit. */
  maxConcurrent?: number;
  /**
   * How long a ready request waits for company before the queue is released. Default 2 ms for explicit-batch
   * backends, which pack what arrives together, and 0 (the next timer tick) otherwise: continuous servers batch
   * by themselves, and the window only decides ordering when requests queue.
   */
  coalesceMs?: number;
  /** Serve turns of running calls before new calls. Default true. */
  priority?: boolean;
  /** Send requests with the same prompt prefix next to each other. Default true except in serial mode. */
  adjacency?: boolean;
  /** Time source, for tests. */
  now?: () => number;
};

export type OccupancySummary = {
  requests: number; batches: number; maxInFlight: number; meanInFlight: number; meanBatchSize: number;
  queueWaitMs: { mean: number; p50: number; p95: number; max: number };
  running: number; fresh: number;
};

export type Scheduler = {
  readonly capabilities: BackendCapabilities;
  /** Wait for a slot. Resolves with a release function; rejects if `signal` aborts or the request is cancelled. */
  acquire(meta?: ChatRequestMeta, signal?: AbortSignal): Promise<() => void>;
  /** A transport whose requests wait here. A streamed reply keeps its slot until it has been read to the end. */
  transport(transport: ChatTransport): ChatTransport;
  /** The scheduler as a plain request limit (no priority information), for code written against `RequestLimit`. */
  asRequestLimit(): RequestLimit;
  /** Change the concurrency, for example once a managed server has said how many slots it started. */
  setMaxConcurrent(size: number): void;
  /** Reject the queued requests of one invocation. Requests already sent run to their end. */
  cancel(invocationId: string, reason?: unknown): number;
  /** Reject every queued request. */
  close(reason?: unknown): void;
  readonly queued: number;
  readonly inFlight: number;
  occupancy(): OccupancySummary;
  /** Recent request records, newest last. */
  records(): readonly ScheduleInfo[];
};

type Waiter = { meta: ChatRequestMeta; signal?: AbortSignal; enqueued: number; priority: 'running' | 'new'; group: string;
  seq: number; grant: (release: () => void) => void; reject: (reason: unknown) => void; detach: () => void };

const MAX_RECORDS = 4096, MAX_REMEMBERED_INVOCATIONS = 10_000;

/** A cheap stable key of a request's shared prefix: its leading system message. */
export function prefixKey(body: Json | undefined): string {
  const messages = body?.messages;
  if (!Array.isArray(messages)) return '';
  const first = messages[0] as { content?: unknown } | undefined;
  const content = typeof first?.content === 'string' ? first.content : JSON.stringify(first?.content ?? '');
  let hash = 5381;
  for (let index = 0; index < content.length; index++) hash = ((hash << 5) + hash + content.charCodeAt(index)) | 0;
  return `${content.length}:${hash >>> 0}`;
}

export function createScheduler(options: SchedulerOptions = {}): Scheduler {
  const mode = options.mode ?? 'server-continuous';
  let maxConcurrent = options.maxConcurrent ?? Infinity;
  if (!(maxConcurrent >= 1)) throw new RangeError('a scheduler needs maxConcurrent of at least 1');
  const coalesceMs = options.coalesceMs ?? (mode === 'explicit-batch' ? 2 : 0);
  const usePriority = options.priority ?? true, useAdjacency = options.adjacency ?? mode !== 'serial';
  const now = options.now ?? (() => performance.now());
  let active = 0, sequence = 0, batchCounter = 0, closed = false;
  const queue: Waiter[] = [];
  const seen = new Map<string, number>();   // invocation id -> requests so far (insertion order = age)
  const records: ScheduleInfo[] = [];
  let timer: ReturnType<typeof setTimeout> | undefined, flushing = false;
  const groupOrder = new Map<string, number>();

  const take = (): Waiter | undefined => {
    if (!queue.length) return undefined;
    let best = 0;
    if (usePriority || useAdjacency) {
      for (let index = 1; index < queue.length; index++) {
        const a = queue[best]!, b = queue[index]!;
        if (usePriority && a.priority !== b.priority) { if (b.priority === 'running') best = index; continue; }
        if (useAdjacency && a.group !== b.group) {
          const ga = groupOrder.get(a.group) ?? Infinity, gb = groupOrder.get(b.group) ?? Infinity;
          if (gb < ga) best = index;
        }
      }
    }
    return queue.splice(best, 1)[0];
  };

  const dispatch = (): void => {
    if (!queue.length || active >= maxConcurrent) return;
    const room = Math.min(queue.length, maxConcurrent - active);
    const batchId = `b${++batchCounter}`;
    const started = now();
    const batch: Waiter[] = [];
    for (let index = 0; index < room; index++) { const next = take(); if (next) batch.push(next); }
    for (const waiter of batch) {
      waiter.detach();
      active++;
      let held = true;
      const release = () => { if (!held) return; held = false; active--; afterRelease(); };
      const info: ScheduleInfo = { batch_id: batchId, batch_size: batch.length, in_flight: active,
        queue_wait_ms: Math.max(0, Math.round((started - waiter.enqueued) * 100) / 100), priority: waiter.priority };
      records.push(info); if (records.length > MAX_RECORDS) records.shift();
      try { waiter.meta.onScheduled?.(info); } catch { /* a reporting hook must not stall a request */ }
      waiter.grant(release);
    }
  };

  const schedule = (): void => {
    if (timer !== undefined || flushing || !queue.length) return;
    if (active >= maxConcurrent) return;   // a release will dispatch
    timer = setTimeout(() => { timer = undefined; dispatch(); }, coalesceMs);
  };
  // A freed slot is refilled at once, from the queue in priority order: no window, requests are already waiting.
  const afterRelease = (): void => {
    flushing = true;
    try { if (timer !== undefined) { clearTimeout(timer); timer = undefined; } dispatch(); }
    finally { flushing = false; }
    schedule();
  };

  const remember = (id: string): number => {
    const count = (seen.get(id) ?? 0) + 1;
    seen.delete(id); seen.set(id, count);
    if (seen.size > MAX_REMEMBERED_INVOCATIONS) seen.delete(seen.keys().next().value!);
    return count;
  };

  const percentile = (sorted: number[], p: number) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)]! : 0;
  const scheduler: Scheduler = {
    capabilities: { batching: { get mode() { return mode; }, get maxConcurrent() { return maxConcurrent; } } },
    acquire(meta = {}, signal) {
      if (closed) return Promise.reject(new Error('model scheduler is closed'));
      try { signal?.throwIfAborted(); } catch (error) { return Promise.reject(error); }
      const turn = meta.invocation_id ? remember(meta.invocation_id) : 1;
      const priority: Waiter['priority'] = turn > 1 || (meta.retry ?? 0) > 0 || (meta.turn ?? 1) > 1 ? 'running' : 'new';
      const group = meta.group ?? '';
      if (!groupOrder.has(group)) { groupOrder.set(group, groupOrder.size); if (groupOrder.size > 1024) groupOrder.delete(groupOrder.keys().next().value!); }
      return new Promise<() => void>((grant, reject) => {
        const waiter: Waiter = { meta, signal, enqueued: now(), priority, group, seq: sequence++, grant, reject, detach: () => {} };
        const abort = () => {
          const at = queue.indexOf(waiter);
          if (at >= 0) queue.splice(at, 1);
          reject(signal!.reason);
        };
        waiter.detach = () => signal?.removeEventListener('abort', abort);
        signal?.addEventListener('abort', abort, { once: true });
        queue.push(waiter);
        schedule();
      });
    },
    transport(transport) {
      return async (body, signal, meta = {}) => {
        const release = await scheduler.acquire({ ...meta, group: meta.group ?? prefixKey(body) }, signal);
        let result;
        try { result = await transport(body, signal, meta); } catch (error) { release(); throw error; }
        const stream = result as AsyncIterable<Json>;
        if (typeof stream[Symbol.asyncIterator] !== 'function') { release(); return result; }
        return (async function* () { try { yield* stream; } finally { release(); } })();
      };
    },
    asRequestLimit() {
      return { get size() { return maxConcurrent; }, acquire: signal => scheduler.acquire({}, signal) };
    },
    setMaxConcurrent(size) {
      if (!(size >= 1)) throw new RangeError('a scheduler needs maxConcurrent of at least 1');
      maxConcurrent = size;
      schedule();
    },
    cancel(invocationId, reason = new Error('model request cancelled')) {
      let count = 0;
      for (let index = queue.length - 1; index >= 0; index--) {
        const waiter = queue[index]!;
        if (waiter.meta.invocation_id !== invocationId) continue;
        queue.splice(index, 1); waiter.detach(); waiter.reject(reason); count++;
      }
      seen.delete(invocationId);
      return count;
    },
    close(reason = new Error('model scheduler is closed')) {
      closed = true;
      if (timer !== undefined) { clearTimeout(timer); timer = undefined; }
      for (const waiter of queue.splice(0)) { waiter.detach(); waiter.reject(reason); }
    },
    get queued() { return queue.length; },
    get inFlight() { return active; },
    occupancy() {
      const waits = records.map(item => item.queue_wait_ms).sort((a, b) => a - b);
      const batches = new Set(records.map(item => item.batch_id)).size;
      const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
      const count = records.length || 1;
      return { requests: records.length, batches, maxInFlight: Math.max(0, ...records.map(item => item.in_flight)),
        meanInFlight: sum(records.map(item => item.in_flight)) / count, meanBatchSize: sum(records.map(item => item.batch_size)) / count,
        queueWaitMs: { mean: sum(waits) / count, p50: percentile(waits, 0.5), p95: percentile(waits, 0.95), max: waits.at(-1) ?? 0 },
        running: records.filter(item => item.priority === 'running').length,
        fresh: records.filter(item => item.priority === 'new').length };
    },
    records() { return records; },
  };
  return scheduler;
}

/** The scheduler for a backend, from a profile's `concurrency` and `batching` settings. */
export function schedulerForSettings(settings: { concurrency?: number;
  batching?: { mode?: BatchingMode; maxConcurrent?: number; coalesceMs?: number; priority?: boolean } },
  fallback: { maxConcurrent?: number } = {}): Scheduler {
  const batching = settings.batching ?? {};
  return createScheduler({ mode: batching.mode, coalesceMs: batching.coalesceMs, priority: batching.priority,
    maxConcurrent: batching.maxConcurrent ?? settings.concurrency ?? fallback.maxConcurrent });
}
