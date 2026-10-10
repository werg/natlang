/**
 * The messages between the page's BrowserCallStore (call-store.ts) and the call store's worker (call-store-worker.ts).
 * A module of its own so the page's side and the Node build type against it without compiling the worker entry, which
 * is only ever a browser bundle (build-browser.mjs).
 */
import type { CallStore } from '../calls/store-core.js';
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
