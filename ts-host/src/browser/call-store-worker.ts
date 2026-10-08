/**
 * The browser's call store (§3.4): a dedicated worker that keeps the store's SQLite database in the origin private
 * file system, through sqlite-wasm's OPFS SAH-pool VFS (no cross-origin isolation headers needed). The page talks to
 * it through BrowserCallStore (call-store.ts): writes are posted in order, reads are answered by message, and the
 * worker pushes the current compilations whenever they change, so dispatch on the page stays synchronous.
 */
import sqlite3InitModule from '@sqlite.org/sqlite-wasm';
import type { CallStore } from '../calls/store-core.js';
import { wasmCallStore, type DatabaseMedium, type WasmDb } from '../calls/store-wasm.js';
import type { SiteStatistics } from '../runtime/iterate.js';

/** Store methods the page may call by name (reads and the store's own maintenance). */
export const QUERY_METHODS = ['calls', 'call', 'value', 'blob', 'events', 'children', 'hot', 'annotations', 'compilations', 'compilation', 'caseCalls',
  'callCases', 'caseStats', 'declines', 'jobs', 'pendingJobs', 'settings', 'writeSettings', 'pin', 'setTier', 'setCompilationStatus',
  'evict', 'totalBytes', 'diskBytes', 'countCalls', 'agentCalls'] as const;
export type QueryMethod = typeof QUERY_METHODS[number];

export type ToWorker =
  | { op: 'open'; id: number; name: string; mode?: string }
  | { op: 'write'; method: 'record' | 'begin' | 'progress' | 'caseServed' | 'enqueue' | 'annotate'; args: unknown[] }
  | { op: 'statistics'; action: 'write' | 'reset'; key?: string; value?: SiteStatistics; prefix?: string }
  | { op: 'query'; id: number; method: QueryMethod; args: unknown[] }
  | { op: 'close'; id: number };
export type Snapshot = { version: number; settings: ReturnType<CallStore['settings']>;
  compilations: Record<string, NonNullable<ReturnType<CallStore['currentCompilation']>>> };
export type FromWorker =
  | { type: 'reply'; id: number; ok: true; value?: unknown; snapshot?: Snapshot; statistics?: Record<string, SiteStatistics> }
  | { type: 'reply'; id: number; ok: false; error: string }
  | { type: 'snapshot'; snapshot: Snapshot }
  | { type: 'error'; error: string };

const scope = self as unknown as { postMessage(message: FromWorker): void; onmessage: ((event: MessageEvent<ToWorker>) => void) | null;
  navigator: Navigator };
let store: CallStore | undefined;
let medium: DatabaseMedium | undefined;
let sent = -1;

function snapshot(): Snapshot {
  const current = store!.compilations({ status: 'current', limit: 10_000 });
  const compilations: Snapshot['compilations'] = {};
  for (const row of current) { const full = store!.compilation(row.id); if (full) compilations[row.definition_key] = full; }
  return { version: store!.version(), settings: store!.settings(), compilations };
}
/** Push the compilations to the page when they changed. */
function publish(): void {
  const version = store!.version();
  if (version === sent) return;
  sent = version;
  scope.postMessage({ type: 'snapshot', snapshot: snapshot() });
}
async function estimate(): Promise<void> {
  try {
    const { quota, usage } = await scope.navigator.storage.estimate();
    if (medium && quota !== undefined) medium.free = Math.max(0, quota - (usage ?? 0));
  } catch { /* unknown */ }
}

async function open(name: string, mode?: string): Promise<Snapshot> {
  const sqlite3 = await sqlite3InitModule();
  const pool = await sqlite3.installOpfsSAHPoolVfs({ name: `natlang-${name}`, initialCapacity: 6 });
  const db = new pool.OpfsSAHPoolDb(`/${name}.sqlite`) as unknown as WasmDb;
  // The SAH-pool VFS has no shared memory, so no WAL; the database is this worker's alone.
  db.exec({ sql: 'PRAGMA journal_mode = TRUNCATE; PRAGMA synchronous = NORMAL;' });
  ({ store, medium } = wasmCallStore(`opfs:${name}`, db, { mode: () => mode }));
  await estimate();
  setInterval(() => { void estimate(); }, 60_000);
  const value = snapshot();
  sent = value.version;
  return value;
}

// Messages are handled one at a time, in order, after the database is open.
let queue: Promise<void> = Promise.resolve();
scope.onmessage = event => { const message = event.data; queue = queue.then(() => handle(message)); };

async function handle(message: ToWorker): Promise<void> {
  try {
    if (message.op === 'open') {
      const value = await open(message.name, message.mode);
      scope.postMessage({ type: 'reply', id: message.id, ok: true, snapshot: value, statistics: store!.iterationStatistics().export?.() as Record<string, SiteStatistics> });
      return;
    }
    if (!store) {
      if ('id' in message) scope.postMessage({ type: 'reply', id: message.id, ok: false, error: 'the call store is not open' });
      return;
    }
    if (message.op === 'write') {
      (store[message.method] as (...args: unknown[]) => void)(...message.args);
      publish();
    } else if (message.op === 'statistics') {
      const statistics = store.iterationStatistics();
      if (message.action === 'write') void statistics.write(message.key!, message.value!);
      else void statistics.reset?.(message.prefix);
    } else if (message.op === 'query') {
      if (!(QUERY_METHODS as readonly string[]).includes(message.method)) throw new Error(`not a store query: ${message.method}`);
      const value = (store[message.method] as (...args: unknown[]) => unknown)(...message.args);
      scope.postMessage({ type: 'reply', id: message.id, ok: true, value });
      publish();
    } else if (message.op === 'close') {
      store.close(); store = undefined;
      scope.postMessage({ type: 'reply', id: message.id, ok: true });
    }
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    if ('id' in message) scope.postMessage({ type: 'reply', id: message.id, ok: false, error: text });
    else scope.postMessage({ type: 'error', error: text });
  }
}
