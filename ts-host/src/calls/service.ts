/**
 * The call store as a typed host service (§3.5): natlang programs (the specializer, the improver, an application's own
 * diagnosis) read records through `natlang:services` like any other capability. Values come back inlined. Node only.
 */
import type { CallStore, CallSummary, HotDefinition } from './store.js';
import type { CaseRole, CaseStats } from './types.js';

export type TraceCall = { call: string; definition: string; executor: string; outcome: string; started: string; ms: number;
  inputs: Record<string, unknown>; result: unknown; effects: { call: string; args: unknown; result?: unknown; error?: string }[];
  evals: string[]; case?: string };

/** The service object. Every method is synchronous and returns plain data. */
export function tracesService(store: CallStore) {
  const call = (id: string): TraceCall | null => {
    const record = store.call(id);
    if (!record) return null;
    return { call: record.call_id, definition: record.definition.name, executor: record.executor.kind, outcome: record.outcome,
      started: record.started_at, ms: record.cost.wall_ms,
      inputs: Object.fromEntries(Object.entries({ ...record.captures, ...record.inputs }).map(([name, ref]) => [name, store.value(ref) ?? null])),
      result: store.value(record.output) ?? null,
      effects: record.effects.map(effect => ({ call: `${effect.service}.${effect.method}`, args: store.value(effect.args) ?? null,
        ...(effect.error ? { error: effect.error } : { result: store.value(effect.result) ?? null }) })),
      evals: record.approach.evals, ...(record.executor.case_hash ? { case: record.executor.case_hash } : {}) };
  };
  return {
    hot: (options: { program?: string; since?: string; by?: 'calls' | 'tokens' | 'wall_ms'; limit?: number } = {}): HotDefinition[] =>
      store.hot({ ...options, limit: Math.min(options.limit ?? 20, 200) }),
    calls: (filter: { definition?: string; outcome?: string; executor?: string; since?: string; limit?: number; after?: string } = {}): CallSummary[] =>
      store.calls({ ...filter, limit: Math.min(filter.limit ?? 20, 200) }),
    call,
    children: (id: string): CallSummary[] => store.children(id),
    annotate: (id: string, kind: string, value: unknown): void => store.annotate(id, kind, value, 'traces-service'),
    compilation: (definition: string): { id: string; cases: CaseStats[]; source: string } | null => {
      const row = store.compilations({ definition, status: 'current', limit: 1 })[0];
      const compilation = row && store.compilation(row.id);
      return compilation ? { id: compilation.id, cases: compilation.cases, source: compilation.files['cases.ts'] ?? '' } : null;
    },
    caseCalls: (hash: string, role?: CaseRole, limit = 20) => store.caseCalls(hash, role, Math.min(limit, 200)),
  };
}

/** What the model is shown of the service (`serviceDeclarations.traces`). */
export const TRACES_DECLARATIONS = `/** A recorded call, with its values. */
export type TraceCall = { call: string; definition: string; executor: 'agent' | 'crisp' | 'crisp-agent'; outcome: string; started: string; ms: number;
  inputs: Record<string, unknown>; result: unknown; effects: { call: string; args: unknown; result?: unknown; error?: string }[];
  /** The eval programs the executor ran, in order. */ evals: string[]; case?: string };
export type CallSummary = { call_id: string; definition_name: string; executor: string; outcome: string; started_at: string; wall_ms: number;
  approach_hash: string | null };
/** Functions by how often (or how expensively) they ran. */
export function hot(options?: { program?: string; since?: string; by?: 'calls' | 'tokens' | 'wall_ms'; limit?: number }): { definition_name: string;
  definition_source: string | null; definition_key: string; calls: number; agent_calls: number; crisp_calls: number; tokens: number }[];
/** Recorded calls, newest first. definition is a function name, source path or revision key. */
export function calls(filter?: { definition?: string; outcome?: string; executor?: string; since?: string; limit?: number; after?: string }): CallSummary[];
/** One recorded call with its inputs, result, service calls and eval programs; null when it is not in the store. */
export function call(id: string): TraceCall | null;
/** The calls a call started. */
export function children(id: string): CallSummary[];
/** Attach feedback or a judgment to a call. */
export function annotate(id: string, kind: string, value: unknown): void;
/** The current compilation of a function: its cases with their numbers and its cases.ts source. */
export function compilation(definition: string): { id: string; cases: unknown[]; source: string } | null;
/** The calls linked to a case, by role (training, held-out, served, handed-off, shadow, audit). */
export function caseCalls(hash: string, role?: string, limit?: number): { call_id: string; role: string; verdict: string | null }[];
`;
