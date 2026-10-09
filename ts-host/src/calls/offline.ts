/**
 * Offline work over the store (§6.2, §6.3): shadow replays of cases against agent calls, audits of crisp calls through
 * the agent, and rebuilding definitions from their recorded program. Runs in a runtime of its own (the specializer
 * loop), never on a live call's path. Node only.
 */
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { natlangDefinition } from '../runtime/callable.js';
import { loadNamedFunction } from '../runtime/loader.js';
import { nodeSourceFiles } from '../runtime/node-files.js';
import { invokeDefinition, type CallableDefinition, type CaptureCell } from '../runtime/kernel.js';
import { currentFrame } from '../runtime/context.js';
import type { NatlangRuntime } from '../runtime/runtime.js';
import { loadCases, type LoadedCase } from './compilations.js';
import { judge, recordedBehavior, type Behavior } from './judge.js';
import { definitionKey } from './recorder.js';
import { recordedArguments, replay, subtreeEffects } from './replay.js';
import type { AuditJob, CallStore } from './store.js';
import { isModelEvidence, type CallRecord, type Verdict } from './types.js';

/** The definition a record ran: loaded from its program when the source still matches, else rebuilt from the record. */
export function definitionFor(store: CallStore, record: CallRecord): { definition: CallableDefinition; loaded: boolean } {
  const identity = record.definition;
  if (identity.source?.endsWith('.nl') && record.program_root) {
    const path = resolve(record.program_root, identity.source);
    if (existsSync(path)) {
      try {
        const definition = natlangDefinition(loadNamedFunction(path, nodeSourceFiles(record.program_root)));
        if (definitionKey(definition) === identity.key) return { definition, loaded: true };
      } catch { /* fall back to the record */ }
    }
  }
  const body = store.value(identity.instructions);
  if (typeof body !== 'string') throw new Error(`the instructions of ${identity.name} were not recorded`);
  return { loaded: false, definition: { id: identity.id, name: identity.name, body, params: identity.params, returns: identity.returns,
    types: identity.types, codebase: {}, subtype: identity.subtype, ...(identity.readout ? { readout: identity.readout as 'decision' } : {}),
    ...(identity.source ? { source: identity.source } : {}) } };
}

export const signatureOf = (definition: { name: string; params: { name: string; type: string; optional?: boolean }[]; returns: string }): string =>
  `${definition.name}(${definition.params.map(param => `${param.name}${param.optional ? '?' : ''}: ${param.type}`).join(', ')}): ${definition.returns}`;

/** Find a case by hash in the current compilation of a definition revision, loaded in `definition`'s context. */
export function caseFor(store: CallStore, definitionKeyValue: string, hash: string, definition: CallableDefinition): LoadedCase | undefined {
  const compilation = store.currentCompilation(definitionKeyValue);
  if (!compilation?.files['cases.ts']) return undefined;
  return loadCases(compilation.id, compilation.files['cases.ts'], definition.codebase, definition.types).find(item => item.hash === hash);
}

/** Run a case on a recorded call's inputs against its recorded effects; its behavior, or why it could not run. */
export async function replayCaseOn(runtime: NatlangRuntime, store: CallStore, item: Pick<LoadedCase, 'run'>, record: CallRecord):
  Promise<Behavior | { skipped: string }> {
  const args = recordedArguments(store, record);
  if (!args) return { skipped: 'the call has inputs that were not recorded exactly' };
  const outcome = await replay(runtime, item.run, args, subtreeEffects(store, record),
    { captureNames: Object.keys(record.captures) });
  return { ...(outcome.error !== undefined ? { error: outcome.error } : { value: outcome.value }), effects: outcome.observed,
    captureWrites: outcome.captureWrites, files: outcome.files ?? [] };
}

/** Run the agent on a recorded call's inputs, answering services from `effects` (an audit, §6.3). */
export async function rerunAgent(runtime: NatlangRuntime, store: CallStore, definition: CallableDefinition, record: CallRecord):
  Promise<Behavior | { skipped: string }> {
  const args = recordedArguments(store, record);
  if (!args) return { skipped: 'the call has inputs that were not recorded exactly' };
  const captures: Record<string, CaptureCell> = Object.fromEntries(Object.keys(record.captures).map(name => {
    let value = args[name];
    return [name, { name, type: 'unknown', mutable: true, get: () => value, set: (next: unknown) => { value = next; args[name] = next; } }];
  }));
  const positional = [...(definition.subtype === 'directory-reducer' ? [args.folder] : []), ...definition.params.map(param => args[param.name])];
  while (positional.length && positional.at(-1) === undefined) positional.pop();
  const outcome = await replay(runtime, () => invokeDefinition(currentFrame()!, definition, positional,
    Object.keys(captures).length ? { captures } : {}), args, subtreeEffects(store, record),
    { captureNames: Object.keys(record.captures), taskOptions: { auditOf: record.call_id } });
  return { ...(outcome.error !== undefined ? { error: outcome.error } : { value: outcome.value }), effects: outcome.observed,
    captureWrites: outcome.captureWrites, files: outcome.files ?? [] };
}

/** Judge a case's behavior against the agent's on one call. */
export async function verdictFor(runtime: NatlangRuntime, store: CallStore, definition: CallableDefinition, record: CallRecord,
  agent: Behavior, crisp: Behavior): Promise<{ verdict: Verdict; differences: string[] }> {
  const inputs = Object.fromEntries(Object.entries({ ...record.captures, ...record.inputs }).map(([name, ref]) => [name, store.value(ref)]));
  return judge(runtime, { instructions: definition.body, signature: signatureOf(definition), inputs, reference: agent, candidate: crisp,
    annotations: store.annotations(record.call_id).map(item => ({ kind: item.kind, value: item.value })), seed: record.call_id });
}

/** Process one shadow or audit job and record its verdict. */
export async function runJob(runtime: NatlangRuntime, store: CallStore, job: AuditJob): Promise<{ status: 'done' | 'skipped' | 'failed'; verdict?: Verdict; detail?: string }> {
  const record = store.call(job.call_id);
  if (!record) return { status: 'skipped', detail: 'the call is no longer in the store' };
  const { definition } = definitionFor(store, record);
  const item = caseFor(store, record.definition.key, job.case_hash, definition);
  if (!item) return { status: 'skipped', detail: 'the case is no longer in the current compilation' };
  if (job.kind === 'shadow') {
    if (!isModelEvidence(record)) return { status: 'skipped', detail: 'the call did not run a declared model (a scripted or undeclared executor)' };
    const crisp = await replayCaseOn(runtime, store, item, record);
    if ('skipped' in crisp) return { status: 'skipped', detail: crisp.skipped };
    const { verdict, differences } = await verdictFor(runtime, store, definition, record, recordedBehavior(store, record), crisp);
    store.caseVerdict(job.case_hash, record.call_id, 'shadow', verdict);
    if (verdict === 'better') store.finding({ definitionKey: record.definition.key, definitionId: record.definition.id, definitionName: record.definition.name,
      definitionSource: record.definition.source, kind: 'executor-worse', summary: `live calls where case ${job.case_hash} did better than the executor`,
      detail: { case: job.case_hash, call: record.call_id, differences } });
    return { status: 'done', verdict, detail: differences.join('; ') };
  }
  const agent = await rerunAgent(runtime, store, definition, record);
  if ('skipped' in agent) return { status: 'skipped', detail: agent.skipped };
  const { verdict, differences } = await verdictFor(runtime, store, definition, record, agent, recordedBehavior(store, record));
  store.caseVerdict(job.case_hash, record.call_id, 'audit', verdict);
  return { status: 'done', verdict, detail: differences.join('; ') };
}

/** Process pending jobs, oldest first. Returns how many ran. */
export async function runJobs(runtime: NatlangRuntime, store: CallStore, options: { limit?: number; signal?: AbortSignal;
  log?: (line: string) => void } = {}): Promise<number> {
  let count = 0;
  for (const job of store.pendingJobs(options.limit ?? 50)) {
    if (options.signal?.aborted) break;
    try {
      const result = await runJob(runtime, store, job);
      store.finishJob(job.id, result.status, result.verdict, result.detail);
      options.log?.(`${job.kind} ${job.case_hash} on ${job.call_id}: ${result.status}${result.verdict ? ` ${result.verdict}` : ''}${result.detail ? ` (${result.detail})` : ''}`);
    } catch (error) {
      store.finishJob(job.id, 'failed', null, error instanceof Error ? error.message : String(error));
      options.log?.(`${job.kind} ${job.case_hash} on ${job.call_id}: failed: ${error instanceof Error ? error.message : String(error)}`);
    }
    count++;
  }
  return count;
}
