/**
 * Call records (`natlang.calls/1`, plans/TRACE_SPECIALIZATION.md §3): what the runtime keeps of every natlang call so
 * calls can be queried, replayed, and compiled into crisp cases. Values are content-addressed; a record names them by
 * hash and the store holds the bytes.
 */
export const CALLS_VERSION = 'natlang.calls/1';

/** A recorded value: its exact JSON (by hash) when it is portable and within the per-value bound, else what is known. */
export type ValueRef =
  | { complete: true; hash: string; bytes: number }
  | { complete: false; reason: 'oversize' | 'nonportable' | 'excluded'; hash?: string; bytes?: number; type?: string; preview?: string };

/** One service call made during a call (by the agent's eval code, or by a crisp case). */
export type EffectRecord = { order: number; service: string; method: string; args: ValueRef; result?: ValueRef;
  /** Whether the method returned a promise, so a replay answers the same way. */
  async?: boolean; error?: string; by: 'agent' | 'crisp' };

/** Which executor produced a call's result. `crisp-agent`: a case failed and the agent finished the call. */
export type ExecutorKind = 'agent' | 'crisp' | 'crisp-agent';

export type SiteKind = 'named' | 'inline' | 'iterate' | 'internal';

/** The identity of what ran: `key` is the definition revision compilations are keyed by (§7.3). */
export type DefinitionIdentity = { id: string; name: string; source: string | null; key: string; interface: string;
  site: SiteKind; template?: string; subtype: 'function' | 'directory-reducer'; params: { name: string; type: string; optional?: boolean }[];
  returns: string;
  /** The instructions as run, and the type aliases in scope: enough to rebuild the definition offline when its source cannot be loaded. */
  instructions: ValueRef; types: Record<string, string>; readout?: string };

export type CallCost = { model_requests: number; tokens_in: number; tokens_out: number; wall_ms: number; turns: number; evals: number };

export type FolderRecord = { mode: 'apply' | 'direct'; changes: { path: string; kind: string; after?: ValueRef }[] };

export type CallRecord = {
  version: typeof CALLS_VERSION;
  call_id: string; parent_call_id: string | null;
  /** Which eval of the parent call was running when this call started (0-based), when a parent eval started it. */
  parent_action_index: number | null;
  task_id: string; program_id: string | null; build_hash: string | null;
  /** The program's directory on this machine, when known: where offline work reloads the definition from. */
  program_root: string | null;
  definition: DefinitionIdentity;
  executor: { kind: ExecutorKind; model_id: string | null; model_revision: string | null; case_hash?: string;
    /** For `crisp-agent`: why the case stopped. */
    case_error?: string };
  inputs: Record<string, ValueRef>;
  captures: Record<string, ValueRef>;
  capture_writes: { name: string; after: ValueRef }[];
  output: ValueRef | null;
  outcome: string; detail: string;
  started_at: string; ended_at: string;
  effects: EffectRecord[];
  folder: FolderRecord | null;
  /** The eval programs the agent ran, in order (successful and failed), and the hash of their normalized form. */
  approach: { evals: string[]; hash: string | null };
  cost: CallCost;
  /** Typed features of the inputs (§5.2), computed when the call is recorded. */
  features: Record<string, string | number | boolean>;
  /** Set on audit and shadow runs: the call they re-ran or checked. */
  audit_of?: string | null;
  events: ValueRef | null;
};

/** Machine settings (`<store>/config.json`, environment overrides). */
export type CallStoreSettings = {
  /** Largest value recorded exactly, as JSON bytes. */
  maxValueBytes: number;
  /** Store size at which eviction starts. */
  maxStoreBytes: number;
  /** How far compilations may go on this machine. */
  specialization: 'off' | 'shadow' | 'on';
  /** Fraction of crisp-served calls queued for an offline audit through the agent. */
  auditRate: number;
  /** Largest share of `worse` verdicts (and of hand-offs) a case may have and stay accepted or active. */
  acceptanceBound: number;
  /** Comparisons a case needs (held-out replays plus shadow checks) before it is promoted to active. */
  promotionComparisons: number;
  /** Calls of one definition revision before the specializer looks at it. */
  minCalls: number;
};

export const DEFAULT_SETTINGS: CallStoreSettings = { maxValueBytes: 1 << 20, maxStoreBytes: 50 * 2 ** 30, specialization: 'on',
  auditRate: 0.05, acceptanceBound: 0.05, promotionComparisons: 10, minCalls: 20 };

export type CaseTier = 'shadow' | 'active' | 'demoted' | 'disabled';
export type CaseRole = 'group' | 'training' | 'held-out' | 'counterexample' | 'shadow' | 'audit' | 'served' | 'handed-off';
export type Verdict = 'equal' | 'better' | 'equivalent' | 'worse' | 'diverged';

export type CaseStats = { hash: string; compilation_id: string; position: number; tier: CaseTier; served: number; handed_off: number;
  compared: number; worse: number; better: number; audited: number; audit_worse: number; created_at: string;
  promoted_at: string | null; demoted_at: string | null; note: string | null };

export type CompilationRow = { id: string; definition_key: string; definition_id: string; definition_name: string; definition_source: string | null;
  interface_hash: string; program_root: string | null; folder_hash: string; parent_id: string | null; created_at: string;
  status: 'current' | 'superseded' | 'disabled'; files: Record<string, string> };

export type DeclineReason = 'no-clusters' | 'semantic' | 'unstable' | 'effects' | 'not-worth-it';
export type DeclineRow = { definition_key: string; definition_id: string; reason: DeclineReason; why: string; calls_at_decline: number; created_at: string };
