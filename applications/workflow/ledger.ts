/**
 * The `ledger` service: the workflow's checks, read-only. The policy's stages call it to test what they chose
 * before it is applied; the mechanism (service.ts) runs the same checks again when it applies the decision.
 */
import { checkMessage, validate } from './service.js';
import type { Decision, Outgoing, WorkflowEvent, WorkflowState } from './types.js';

export type LedgerService = {
  validate(state: WorkflowState, event: WorkflowEvent, decision: Decision): string | null;
  checkMessage(state: WorkflowState, message: Outgoing): string | null;
};

export function ledgerService(): LedgerService {
  return { validate, checkMessage };
}

/** What the model is told about the service. */
export const ledgerDeclaration = `/** The workflow's checks. Each returns null when the choice is allowed, otherwise one sentence naming what to change. */
export const ledger: {
  /**
   * Is this decision valid for the order in this state on this event? Valid: an order with a pending operation takes
   * reconcile, retry (once a look found no receipt) or wait; an order in phase new reserves, reserved charges (or
   * waits), charged ships, and so on, as the order's phase allows; a cancelled order is compensated or waits.
   */
  validate(state: WorkflowState, event: WorkflowEvent, decision: Decision): string | null;
  /** Is this customer message true of the order and written in plain words, without internal operation keys? */
  checkMessage(state: WorkflowState, message: { kind: string, subject: string, body: string }): string | null;
};`;
