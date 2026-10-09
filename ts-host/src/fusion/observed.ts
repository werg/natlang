/**
 * Observed readers for the fact service (plans/FUSED_PIPELINES.md): the eval programs orchestrating models ran, as the call
 * store recorded them. Only runs that show what a model did count (`isModelEvidence`: a model-driven agent call that spent
 * tokens), and only runs of the orchestrator exactly as it is now (same instructions), so an edit to the prose starts the
 * count again. Node only (the store is SQLite).
 */
import { hexDigest } from '../native/hash.js';
import { isModelEvidence } from '../calls/types.js';
import type { CallStore } from '../calls/store.js';
import type { ObservedRun, ObservedSource } from './facts.js';

/** Runs read per orchestrator, newest first: a bound on the work of a check, not on the evidence a plan may use. */
const RUN_LIMIT = 500;

/** Observed runs from the call store. The revision names the exact runs read (their count and a digest of their IDs). */
export function observedFromStore(store: CallStore, minRuns: number): ObservedSource {
  return { minRuns, runs(scope, instructions) {
    const rows = store.db.prepare(`SELECT call_id FROM calls WHERE (definition_source = ? OR definition_source LIKE ?) AND executor = 'agent'
      AND audit_of IS NULL AND outcome != 'running' ORDER BY started_at DESC, call_id DESC LIMIT ?`)
      .all(scope, `%/${scope}`, RUN_LIMIT) as { call_id: string }[];
    const runs: ObservedRun[] = [];
    for (const { call_id } of rows) {
      const record = store.call(call_id);
      if (!record || !isModelEvidence(record) || !record.approach.evals.length) continue;
      if (String(store.value(record.definition.instructions)).trim() !== instructions.trim()) continue;
      runs.push({ id: call_id, evals: record.approach.evals });
    }
    const ids = runs.map(run => run.id).sort();
    return { revision: `${runs.length}:${hexDigest(ids.join('\n')).slice(0, 16)}`, runs };
  } };
}
