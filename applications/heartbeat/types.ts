import type { Untrusted } from '@natlang/node';

// The hourly check-in as data (plans/HEARTBEAT_PROGRAM.md). Crisp collectors fill the evidence types, the three
// natural-language functions read them and return the decision types, crisp verifiers check the decisions.

export type Machine = "dgx" | "pop";
export type Source = "ledger" | "unit" | "log" | "gpu" | "gate" | "inbox" | "git" | "teacher" | "disk" | "peer-status";

/** A run the owning session declared (watch.json). The owner of the run adds the entry when the job starts. */
export type WatchEntry = {
  run_id: string,
  machine: Machine,
  /** The owning session or agent: the one who decides process control for this run. */
  owner: string,
  /** The systemd unit that runs it. */
  unit: string,
  kind: "training" | "generation" | "eval" | "build" | "service",
  /** "finishes": the run ends by itself; "serves": it should stay active. */
  expected: "finishes" | "serves",
  /** The log the unit writes, or null. */
  log: string | null,
  /** A regular expression that matches a line the run prints when it makes progress. */
  progress_marker: string,
  /** Minutes without a progress line after which the run counts as stalled. */
  stall_after_minutes: number,
  /** Gate report files (JSON) the run writes. */
  gate_paths: string[],
  /** Dotted paths of the fields of those files that say whether a gate passed (for example "token_aligned_reference_passed"). */
  gate_fields: string[],
  /** The document that says what comes after this run, or null. */
  next_steps_doc: string | null,
};

/** One reading of the outside world, as the model sees it. The text is outside text: it is shown as data. */
export type ReadingView = { id: string, source: Source, ok: boolean, text: Untrusted<string> };

export type UnitState = {
  active_state: string,
  sub_state: string,
  result: string,
  exit_status: number | null,
  started_at: string,
};
export type GateReport = { path: string, schema: string, fields: Record<string, string | number | boolean | null> };
export type LedgerClaim = { budget_gb: number, used_gb: number, peak_gb: number, hold_budget: boolean };
export type ErrorCandidate = { reading_id: string, line: Untrusted<string> };

/** Everything the diagnosis of one run reads. */
export type RunEvidence = {
  entry: WatchEntry,
  unit: UnitState,
  /** Minutes since the last log line that matches the entry's progress marker; null when no such line was found. */
  minutes_since_progress: number | null,
  /** The last lines of the log that look like an error, oldest first. */
  error_candidates: ErrorCandidate[],
  gates: GateReport[],
  ledger_claim: LedgerClaim | null,
  headroom_gb: number | null,
  readings: ReadingView[],
};

export type Health = "progressing" | "stalled" | "finished-ok" | "finished-failed" | "blocked" | "gate-failed" | "unknown";
export type Cause = "none" | "admission-refused" | "out-of-memory" | "input-missing" | "code-error" | "gate-failed"
  | "resource-contention" | "stalled-no-error" | "external" | "unknown";
export type Confidence = "high" | "medium" | "low";
export type Citation = { reading_id: string, quote: string };
export type RunDiagnosis = {
  run_id: string,
  health: Health,
  cause: Cause,
  evidence: Citation[],
  summary: string,
  confidence: Confidence,
};

export type CoordMessage = {
  id: string,
  from: string,
  to: string[],
  kind: "note" | "request" | "decision" | "reply" | "close",
  subject: string,
  body: Untrusted<string>,
  reply_to: string | null,
  urgent: boolean,
  sent_at: string,
};
export type WatchedRun = { run_id: string, unit: string };
export type ObligationKind = "answer-request" | "adopt-decision" | "acknowledge" | "inform";
export type Obligation = {
  message_id: string,
  kind: ObligationKind,
  what: string,
  affects: string[],
  reply_needed: boolean,
};
export type InboxTriage = { obligations: Obligation[], digest: string };

export type Resources = {
  headroom_gb: number | null,
  gpu_utilization: number | null,
  teacher_load: number | null,
  /** Units and ledger claims the machine runs that no watch entry declares. */
  unwatched: string[],
};
export type ActionSummary = {
  id: string,
  what: string,
  /** Parameter names with the kind of value each takes. */
  params: Record<string, string>,
  /** The causes this action answers. */
  applies_to: Cause[],
};
export type Proposal = {
  action: string,
  /** A run id, unit name, message id or path. */
  target: string,
  params: Record<string, string>,
  why: string,
  /** Run ids and message ids this proposal answers. */
  cites: string[],
  expected_effect: string,
};
export type CycleMemory = { cycle: string, proposals: Proposal[], agent_actions: { action: string, target: string }[] };
export type Plan = { proposals: Proposal[], idle_resources: string, summary: string };
