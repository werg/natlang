import type { Untrusted } from '@natlang/node';

/** The compact card of one failed student-projection candidate. Crisp code extracts it from the receipt; every text field came from logs or tool output. */
export type FailureCard = {
  family: string, program_id: string,
  error: string,
  /** The receipt's outcome.checks. */
  checks: Record<string, boolean | null>,
  outcome_detail: string,
  /** The model's calls: the first and last turns, each bounded. */
  actions: string[],
  /** The tool feedback after each action, bounded. */
  feedback: string[],
  turn_count: number, delegation_count: number,
};
/** The card as the function reads it: its text fields are quoted data, not instructions. */
export type UntrustedFailureCard = {
  family: string, program_id: string,
  error: Untrusted<string>,
  checks: Record<string, boolean | null>,
  outcome_detail: Untrusted<string>,
  actions: Untrusted<string>[],
  feedback: Untrusted<string>[],
  turn_count: number, delegation_count: number,
};
export type Confidence = "high" | "medium" | "low";
/** tag is one of the given tags, or "new"; proposed_tag names the problem class exactly when tag is "new". */
export type FailureExplanation = {
  tag: string, proposed_tag: string | null, why: string, evidence: { quote: string }[], confidence: Confidence,
};

/** A reason that no admission rule classifies, with how often it occurs and example rows. */
export type BucketEntry = { reason: string, count: number, examples: { id: string, family: string }[] };
export type UntrustedBucketEntry = { reason: Untrusted<string>, count: number, examples: { id: string, family: string }[] };
/** An existing rule category with its action and the reasons it classifies. */
export type RuleCategory = { category: string, action: string, example_reasons: string[] };
export type RuleProposal = { reason: string, category: string, next_action: string, match_prefix: string, rationale: string };
export type RuleProposals = { proposals: RuleProposal[] };

/** The facts of a failed gate, computed by crisp code with the gate's own arithmetic. */
export type GateFacts = {
  schema: string, passed: boolean,
  failed: { stratum: string, metric: string, observed: number, limit: number, direction: "min" | "max", margin: number }[],
  passed_strata: string[],
  worst_windows: { id: string, metric: string, observed: number }[],
  context: { run: string, checkpoint: string | null, step: number | null, earlier_reports: { path: string, passed: boolean }[] },
};
export type GateExplanation = {
  pattern: string,
  candidate_causes: { cause: string, support: string, confidence: Confidence }[],
  next_checks: string[],
  /** Always true: the explanation never changes the gate. */
  gate_unchanged: boolean,
};

/** The result of explainFailure: tag "new" comes with a snake_case proposed_tag and no other tag does. */
export type CheckedFailureExplanation = Is<FailureExplanation, "an explanation whose proposed_tag is a lowercase snake_case name exactly when its tag is new, whose why is one to three sentences, and whose evidence holds at least one non-empty quote">;
/** The result of triageRejections: every proposal's match_prefix is a prefix of its reason. */
export type CheckedRuleProposals = Is<RuleProposals, "proposals each of whose match_prefix is a non-empty prefix of its reason">;
/** The result of explainGateFailure: it states that the gate is unchanged. */
export type CheckedGateExplanation = Is<GateExplanation, "an explanation with gate_unchanged true, a pattern of one to three sentences, at least one candidate cause and at least one next check">;
