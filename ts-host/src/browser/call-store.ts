/**
 * The page's side of the browser call store (call-store-worker.ts). It is a `CallStoreLike` for the runtime: writes go
 * to the worker in order and never wait; settings, the compilations version and the current compilations are kept here
 * from the worker's snapshots, so dispatch reads them synchronously. `query` reads anything else of the store.
 */
import type { CallStoreLike } from '../calls/recorder.js';
import type { CallStore } from '../calls/store-core.js';
import type { CallRecord, CallStoreSettings, CaseStats, CompilationRow } from '../calls/types.js';
import { DEFAULT_SETTINGS } from '../calls/types.js';
import type { IterationStatisticsStore, SiteStatistics } from '../runtime/iterate.js';
import type { FromWorker, QueryMethod, Snapshot, ToWorker } from './call-store-worker.js';

export type BrowserCallStoreOptions = {
  /** The store's name in the origin's private file system (default `calls`). */
  name?: string;
  /** The worker script; by default `call-store-worker.js` beside natlang.js. */
  workerUrl?: string | URL;
  /** Specialization mode for this page ('off' | 'shadow' | 'on'), over the stored setting. */
  specialization?: string;
};

type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };
/** A request to the worker without its ID (distributes over the message kinds). */
type Request<T = Extract<ToWorker, { id: number }>> = T extends unknown ? Omit<T, 'id'> : never;

/** Whether this browser can keep a call store (module workers and the origin private file system). */
export function browserCallStoreAvailable(): boolean {
  return typeof Worker !== 'undefined' && typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function';
}

export class BrowserCallStore implements CallStoreLike {
  /** Open the store and wait until it is ready. */
  static async open(options: BrowserCallStoreOptions = {}): Promise<BrowserCallStore> {
    const store = new BrowserCallStore(options);
    await store.ready;
    return store;
  }

  /** Resolves when the database is open; rejects when it cannot be (for example another tab holds it). */
  readonly ready: Promise<void>;
  private readonly worker: Worker;
  private readonly pending = new Map<number, Pending>();
  private next = 0;
  private failed: string | undefined;
  private state: Snapshot = { version: 0, settings: { ...DEFAULT_SETTINGS }, compilations: {} };
  private readonly statistics = new Map<string, SiteStatistics>();

  /** Starts opening at once; calls made before it is ready are queued in order. */
  constructor(options: BrowserCallStoreOptions = {}) {
    const url = options.workerUrl ?? new URL('./call-store-worker.js', import.meta.url);
    this.worker = new Worker(url, { type: 'module', name: 'natlang-call-store' });
    this.worker.onmessage = event => this.receive(event.data as FromWorker);
    this.worker.onerror = event => this.fail(event.message || 'the call store worker failed');
    const id = ++this.next;
    this.ready = new Promise<unknown>((resolve, reject) => this.pending.set(id, { resolve, reject })).then(() => undefined);
    this.ready.catch(() => { /* reported once by fail() */ });
    this.post({ op: 'open', id, name: options.name ?? 'calls', ...(options.specialization ? { mode: options.specialization } : {}) });
  }

  private post(message: ToWorker): void { if (!this.failed) this.worker.postMessage(message); }
  private fail(error: string): void {
    if (this.failed) return;
    this.failed = error;
    console.warn(`natlang: the browser call store is unavailable; calls are not recorded: ${error}`);
    for (const pending of this.pending.values()) pending.reject(new Error(error));
    this.pending.clear();
    this.state = { ...this.state, settings: { ...this.state.settings, specialization: 'off' } };
  }
  private receive(message: FromWorker): void {
    if (message.type === 'snapshot') { this.state = message.snapshot; return; }
    if (message.type === 'error') { console.warn(`natlang: call store: ${message.error}`); return; }
    const pending = this.pending.get(message.id);
    this.pending.delete(message.id);
    if (!message.ok) {
      if (message.id === 1) this.fail(message.error);
      pending?.reject(new Error(message.error));
      return;
    }
    if (message.snapshot) this.state = message.snapshot;
    if (message.statistics) for (const [key, value] of Object.entries(message.statistics)) this.statistics.set(key, value);
    pending?.resolve(message.value);
  }
  private request(message: Request): Promise<unknown> {
    if (this.failed) return Promise.reject(new Error(this.failed));
    const id = ++this.next;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.post({ ...message, id } as ToWorker); });
  }

  // --- CallStoreLike ----------------------------------------------------------------------------------------------

  settings(): CallStoreSettings { return this.state.settings; }
  version(): number { return this.state.version; }
  currentCompilation(definitionKey: string): (CompilationRow & { cases: CaseStats[] }) | undefined {
    return this.state.compilations[definitionKey];
  }
  record(record: CallRecord, blobs: ReadonlyMap<string, string>, events?: string): void {
    this.post({ op: 'write', method: 'record', args: [record, blobs, events] });
  }
  begin(row: Parameters<CallStore['begin']>[0]): void { this.post({ op: 'write', method: 'begin', args: [row] }); }
  progress(callId: string, events: string): void { this.post({ op: 'write', method: 'progress', args: [callId, events] }); }
  caseServed(caseHash: string, callId: string, handedOff: boolean): void {
    this.post({ op: 'write', method: 'caseServed', args: [caseHash, callId, handedOff] });
  }
  enqueue(kind: 'audit' | 'shadow', caseHash: string, callId: string): void { this.post({ op: 'write', method: 'enqueue', args: [kind, caseHash, callId] }); }
  annotate(callId: string, kind: string, value: unknown, source?: string, pin?: boolean): void {
    this.post({ op: 'write', method: 'annotate', args: [callId, kind, value, source, pin] });
  }
  iterationStatistics(): IterationStatisticsStore {
    return {
      read: key => this.statistics.get(key),
      write: (key, value) => { this.statistics.set(key, value); this.post({ op: 'statistics', action: 'write', key, value }); },
      reset: (prefix = '') => {
        for (const key of [...this.statistics.keys()]) if (key.startsWith(prefix)) this.statistics.delete(key);
        this.post({ op: 'statistics', action: 'reset', prefix });
      },
      export: () => Object.fromEntries(this.statistics),
    };
  }

  // --- Inspection -------------------------------------------------------------------------------------------------

  /**
   * Call a store method in the worker: `await store.query('calls', { definition: 'support', limit: 20 })`,
   * `query('call', id)`, `query('compilation', id)`, `query('hot')`, ... (QUERY_METHODS in call-store-worker.ts).
   */
  query<M extends QueryMethod>(method: M, ...args: Parameters<CallStore[M]>): Promise<ReturnType<CallStore[M]>> {
    return this.request({ op: 'query', method, args }) as Promise<ReturnType<CallStore[M]>>;
  }
  /** Close the database and stop the worker. */
  async close(): Promise<void> {
    try { await this.request({ op: 'close' }); } catch { /* already closed or failed */ }
    this.worker.terminate();
  }
}
