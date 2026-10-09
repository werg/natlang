import type { Is } from '@natlang/node';

/** What a decision can ask the workflow to do next. reconcile and retry act on an operation whose outcome is unknown. */
export type Action = "reserve" | "charge" | "ship" | "refund" | "release" | "reconcile" | "retry" | "wait";

/** A fault the simulated remote injects for one event. lost_ack: the effect happened and its answer was lost. lost_request: the request never arrived. */
export type Fault = "rate_limit" | "definite_failure" | "lost_ack" | "lost_request";

/** What happened to the order. continue: move it forward. cancel: the customer cancelled. reconcile: look again at an unknown outcome. */
export type WorkflowEvent = { kind: "continue" | "cancel" | "reconcile", fault?: Fault };

/** One attempt at a remote effect. The intent is written before the call; done when the remote acknowledged it, failed when it refused (detail says why). */
export type Operation = { key: string, action: Action, status: "intent" | "done" | "failed", detail: string };

/** What a message tells the customer. */
export type MessageKind = "paid" | "shipped" | "delayed" | "problem" | "cancelled";

/** A message to the customer, as the stages write it. */
export type Outgoing = { kind: MessageKind, subject: string, body: string };

/** A queued message. sent is set by whatever delivers the outbox. */
export type OutboxEntry = { key: string, kind: MessageKind, subject: string, body: string, sent: boolean };

/** The durable record of one order. */
export type WorkflowState = {
  order_id: string,
  /** The order's amount, an integer, as the customer was charged. */
  amount: number,
  revision: number,
  /** new, reserved, charged, shipped, refunded, released, charge-failed or shipping-failed. */
  phase: string,
  /** The key of the operation whose outcome is unknown (intent written, no acknowledgement yet); "" when none. */
  pending: string,
  /** How many times the remote was asked about the pending key and had no receipt. 0 when nothing is pending. */
  checks: number,
  /** Open obligations, one sentence each: an unknown outcome, or a compensation that failed. */
  obligations: string[],
  /** Every attempt, oldest first. */
  history: Operation[],
  outbox: OutboxEntry[],
};

/** What to do next, with the reason. */
export type Decision = {
  action: Action,
  reason: string,
  /** When the order should be looked at again if it is still unresolved: milliseconds from now. */
  waitMs?: number,
};

/** The remote's record of an effect it carried out, keyed by the operation key. */
export type Receipt = { key: string, action: Action, order_id: string, amount: number, status: "done" };

/** Everything a decision is made from. receipt is the remote's receipt for state.pending, or null. */
export type Snapshot = { state: WorkflowState, event: WorkflowEvent, receipt: Receipt | null };

/** The policy's numbers. */
export type Limits = {
  /** How many times a transient failure of an operation is retried before the order is compensated. */
  transientRetries: number,
  /** How many looks for a receipt come before the same request is sent again. */
  checksBeforeRetry: number,
  /** Milliseconds until an unresolved operation is looked at again. */
  recheckMs: number,
};

/** Where an order stands. */
export type Situation = "uncertain" | "cancelling" | "owed" | "failed" | "forward" | "settled";

/** Whether the remote's refusal of an operation may succeed on another try. */
export type Failure = "transient" | "definite";

/** One undo. refund returns a charge; release returns the reserved stock. */
export type CompensationStep = { action: "refund" | "release", reason: string };
/** The undos still to do, in order. */
export type Compensation = { steps: CompensationStep[] };

/** An event and what it did, for the customer-communication stages. */
export type Report = { before: WorkflowState, after: WorkflowState, decision: Decision, event: WorkflowEvent };

/** Whether and what to tell the customer about a report. */
export type Moment = "none" | "paid" | "shipped" | "delayed" | "problem" | "cancelled";

/** What the customer may be told, in plain words. */
export type Facts = {
  order: string,
  amount: number,
  /** What happened to the order, in plain words about the order and its goods. */
  summary: string,
  /** What the customer can expect next. */
  next: string,
};

// ---------------------------------------------------------------- refined results
// What a stage returns carries the property its value shows by itself. The ledger keeps everything that needs the order:
// the transition table, the kind of message against the phase, the effects that stand. Crisp code keeps the plain types.
// Each predicate has a crisp checker in refinements.ts.

/** A decision whose wait, when it names one, is a positive whole number of milliseconds. */
export type CheckedDecision = Is<Decision, "a decision whose waitMs, when given, is a positive whole number of milliseconds">;

/** The undos in order: the refund comes before the release, and each is listed once. */
export type CheckedCompensation = Is<Compensation, "steps that list refund before release, each at most once, with a reason for each">;

/** A message with a subject that fits a line and a body. */
export type CheckedOutgoing = Is<Outgoing, "a message whose subject is one line of at most 60 characters without a trailing period, and whose body is not empty">;
