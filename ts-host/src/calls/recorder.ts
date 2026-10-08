/**
 * Recording one call (§3): the kernel opens a `CallCapture` when a call starts, feeds it inputs, effects and the
 * outcome, and the capture writes one `natlang.calls/1` record to the runtime's store when the call ends. Recording
 * never fails a call: a store error is reported once per store and dropped.
 */
import { hexDigest } from '../native/hash.js';
import { inputFeatures, snapshot } from './snapshot.js';
import { approachHash } from './normalize.js';
import { CALLS_VERSION, type CallRecord, type CallStoreSettings, type CaseStats, type CompilationRow, type DefinitionIdentity,
  type EffectRecord, type ExecutorKind, type FolderRecord, type ValueRef } from './types.js';
import type { ItemRecord } from '../runtime/loader.js';

/** What the runtime needs of a call store. `CallStore` (Node) implements it; tests may pass their own. */
export interface CallStoreLike {
  settings(): CallStoreSettings;
  record(record: CallRecord, blobs: ReadonlyMap<string, string>, events?: string): void;
  version(): number;
  currentCompilation(definitionKey: string): (CompilationRow & { cases: CaseStats[] }) | undefined;
  caseServed(caseHash: string, callId: string, handedOff: boolean): void;
  enqueue(kind: 'audit' | 'shadow', caseHash: string, callId: string): void;
}

const reported = new WeakSet<object>();
/** Report a store failure once per store; recording must not fail calls. */
export function reportStoreFailure(store: object, error: unknown): void {
  if (reported.has(store)) return;
  reported.add(store);
  console.warn(`natlang: call recording failed and is skipped for this store: ${error instanceof Error ? error.message : String(error)}`);
}

/** The revision key compilations are stored under: what ran, as run (instructions and contract). */
export function definitionKey(definition: { id: string; body: string; params: unknown; returns: string; types: unknown;
  subtype: string; readout?: string }): string {
  return hexDigest(JSON.stringify({ id: definition.id, body: definition.body, params: definition.params, returns: definition.returns,
    types: definition.types, subtype: definition.subtype, readout: definition.readout ?? null })).slice(0, 32);
}

/** The hash of a context's interface: what a case may call, by name, kind and revision. */
export function interfaceHash(codebase: Record<string, unknown>): string {
  const entries: string[] = [];
  const visit = (level: Record<string, ItemRecord>, prefix: string) => {
    for (const [name, item] of Object.entries(level)) {
      if (!item || typeof item !== 'object' || !('kind' in item)) continue;
      if (item.kind === 'namespace') { visit(item.codebase, `${prefix}${name}.`); continue; }
      entries.push(`${prefix}${name}=${item.kind}@${item.revision}`);
      visit(item.codebase, `${prefix}${name}/`);
    }
  };
  visit(codebase as Record<string, ItemRecord>, '');
  return hexDigest(entries.sort().join('\n')).slice(0, 32);
}

/** Exclusion patterns from `recording.exclude`: a definition source glob, or `definitionName.argument`. */
export function excluded(patterns: readonly string[] | undefined, definition: { name: string; source: string | null }, argument?: string): boolean {
  if (!patterns?.length) return false;
  const glob = (pattern: string) => new RegExp(`^${pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*\*/g, '\u0000')
    .replace(/\*/g, '[^/]*').replace(/\u0000/g, '.*').replace(/\?/g, '.')}$`);
  return patterns.some(pattern => {
    if (definition.source && glob(pattern).test(definition.source)) return true;
    if (argument === undefined) return false;
    return pattern === `${definition.name}.${argument}` || (definition.source !== null && pattern === `${definition.source}:${argument}`);
  });
}

type Pending = { service: string; method: string; args: ValueRef; by: 'agent' | 'crisp'; order: number; seq?: number };

/** The record of one running call. */
export class CallCapture {
  readonly blobs = new Map<string, string>();
  readonly effects: EffectRecord[] = [];
  private readonly pending: Pending[] = [];
  private order = 0;
  private readonly started = Date.now();
  private readonly startedAt = new Date().toISOString();
  inputs: Record<string, ValueRef> = {};
  captures: Record<string, ValueRef> = {};
  captureWrites: { name: string; after: ValueRef }[] = [];
  features: Record<string, string | number | boolean> = {};
  /** Exact host inputs by name, for features and dispatch. */
  hostInputs: Record<string, unknown> = {};
  executor: CallRecord['executor'];
  auditOf: string | null = null;
  folder: FolderRecord | null = null;

  constructor(readonly store: CallStoreLike, readonly settings: CallStoreSettings, readonly base: {
    callId: string; parentCallId: string | null; parentActionIndex: number | null; taskId: string; programId: string | null;
    buildHash: string | null; programRoot: string | null; definition: DefinitionIdentity;
    model: { id: string | null; revision: string | null }; exclude?: readonly string[] }) {
    this.executor = { kind: 'agent', model_id: base.model.id, model_revision: base.model.revision };
  }

  /** Snapshot a value into this record's blobs. */
  ref(value: unknown, excludedValue = false): ValueRef {
    const taken = snapshot(value, this.settings.maxValueBytes, excludedValue);
    if (taken.text !== undefined && taken.ref.complete) this.blobs.set(taken.ref.hash, taken.text);
    return taken.ref;
  }

  setInputs(values: Record<string, unknown>, captures: Record<string, unknown> = {}): void {
    const definition = { name: this.base.definition.name, source: this.base.definition.source };
    this.hostInputs = { ...captures, ...values };
    this.inputs = Object.fromEntries(Object.entries(values).map(([name, value]) =>
      [name, this.ref(value, excluded(this.base.exclude, definition, name))]));
    this.captures = Object.fromEntries(Object.entries(captures).map(([name, value]) =>
      [name, this.ref(value, excluded(this.base.exclude, definition, name))]));
    const visible = Object.fromEntries(Object.entries(this.hostInputs).filter(([name]) => !excluded(this.base.exclude, definition, name)));
    try { this.features = inputFeatures(visible); } catch { this.features = {}; }
  }

  /** One recording-services event (`native/effects.ts`), with its exact arguments and result. */
  effect(event: { phase: 'requested' | 'completed' | 'failed'; service: string; method: string; exact?: { args?: unknown[]; result?: unknown; async?: boolean };
    error?: string; seq?: number }, by: 'agent' | 'crisp' = 'agent'): void {
    if (event.phase === 'requested') {
      this.pending.push({ service: event.service, method: event.method, args: this.ref(event.exact?.args ?? []), by, order: this.order++, seq: event.seq });
      return;
    }
    const index = event.seq !== undefined ? this.pending.findIndex(item => item.seq === event.seq) :
      this.pending.findIndex(item => item.service === event.service && item.method === event.method);
    const started = index >= 0 ? this.pending.splice(index, 1)[0]! :
      { service: event.service, method: event.method, args: this.ref([]), by, order: this.order++ };
    const { seq: _seq, ...effect } = started as Pending;
    this.effects.push({ ...effect, ...(event.phase === 'completed' ? { result: this.ref(event.exact?.result) } : { error: event.error ?? 'failed' }),
      ...(event.exact?.async !== undefined ? { async: event.exact.async } : {}) });
  }

  /** Effects that completed so far, in the order they started. */
  completedEffects(by?: 'agent' | 'crisp'): EffectRecord[] {
    return this.effects.filter(effect => !by || effect.by === by).sort((a, b) => a.order - b.order);
  }

  /** Write the record. Never throws. */
  finish(result: { outcome: string; detail: string; output?: unknown; hasOutput: boolean; events: readonly Record<string, unknown>[] }): CallRecord | undefined {
    try {
      const events = result.events;
      const actions = events.filter(event => event.kind === 'action' && event.name === 'eval').map(event => {
        const code = (event.arguments as Record<string, unknown> | undefined)?.code;
        return { code: typeof code === 'string' ? code : JSON.stringify(code ?? null), ok: !['error', 'rejected'].includes(String(event.outcome)) };
      });
      const evals = actions.map(action => action.code);
      const requests = events.filter(event => event.kind === 'model_request' && event.phase === 'end');
      const definition = this.base.definition;
      const output = result.hasOutput ? this.ref(result.output,
        excluded(this.base.exclude, { name: definition.name, source: definition.source }, 'return')) : null;
      const record: CallRecord = {
        version: CALLS_VERSION, call_id: this.base.callId, parent_call_id: this.base.parentCallId,
        parent_action_index: this.base.parentActionIndex, task_id: this.base.taskId, program_id: this.base.programId,
        build_hash: this.base.buildHash, program_root: this.base.programRoot, definition, executor: this.executor,
        inputs: this.inputs, captures: this.captures, capture_writes: this.captureWrites, output,
        outcome: result.outcome, detail: result.detail.slice(0, 4000), started_at: this.startedAt, ended_at: new Date().toISOString(),
        effects: [...this.effects, ...this.pending.map(({ seq: _seq, ...item }) => ({ ...item, error: 'did not finish' }))].sort((a, b) => a.order - b.order),
        folder: this.folder,
        approach: { evals, hash: this.executor.kind === 'agent' && result.outcome === 'done' ? approachHash(actions.filter(action => action.ok).map(action => action.code), this.hostInputs) : null },
        cost: { model_requests: requests.length, turns: requests.length, evals: evals.length, wall_ms: Date.now() - this.started,
          tokens_in: requests.reduce((sum, event) => sum + (Number(event.prompt_tokens) || 0), 0),
          tokens_out: requests.reduce((sum, event) => sum + (Number(event.completion_tokens) || 0), 0) },
        features: this.features, audit_of: this.auditOf, events: null };
      this.store.record(record, this.blobs, events.length ? events.map(event => JSON.stringify(event)).join('\n') + '\n' : undefined);
      return record;
    } catch (error) { reportStoreFailure(this.store, error); return undefined; }
  }
}

export type { ExecutorKind };
