import type { Action, CompensationStep, Decision, Limits, Operation, Snapshot, WorkflowState } from '../types.js';

const done = (state: WorkflowState, action: Action) => state.history.some(row => row.action === action && row.status === 'done');

/** The undos still to do: a charge that stands is refunded, then reserved stock that stands is released. A shipped order has none. */
function compensation(state: WorkflowState): CompensationStep[] {
  if (done(state, 'ship')) return [];
  const steps: CompensationStep[] = [];
  if (done(state, 'charge') && !done(state, 'refund')) steps.push({ action: 'refund', reason: 'the charge stands and the order is being undone' });
  if (done(state, 'reserve') && !done(state, 'release')) steps.push({ action: 'release', reason: 'the reserved stock stands and the order is being undone' });
  return steps;
}

function compensate(state: WorkflowState, why: string): Decision {
  const step = compensation(state)[0];
  return step ? { action: step.action, reason: `${why}: ${step.reason}` } : { action: 'wait', reason: `${why}: nothing is left to undo` };
}

const transient = (operation: Operation) => operation.detail === 'rate_limit';

/** The same choice as `choose.nl`, as a table: the crisp implementation of the pluggable policy (host: `handle.crisp`). */
export default function crisp(snapshot: Snapshot, limits: Limits): Decision {
  const { state, event, receipt } = snapshot;
  if (state.pending) {
    if (event.kind !== 'reconcile') return { action: 'wait', reason: `${state.pending} has an unknown outcome and is looked at again later`, waitMs: limits.recheckMs };
    if (receipt) return { action: 'reconcile', reason: `the remote has a receipt for ${state.pending}` };
    if (state.checks < limits.checksBeforeRetry) return { action: 'reconcile', reason: `no receipt for ${state.pending} yet; look again`, waitMs: limits.recheckMs };
    return { action: 'retry', reason: `${state.checks} looks found no receipt for ${state.pending}; send it again under the same key` };
  }
  if (event.kind === 'reconcile') return { action: 'wait', reason: 'nothing is pending' };
  if (event.kind === 'cancel') return compensate(state, 'the order was cancelled');
  if (state.obligations.length || state.phase === 'refunded') return compensate(state, 'a compensation is still owed');
  const last = state.history.at(-1);
  if (last?.status === 'failed') {
    const attempts = state.history.filter(row => row.key === last.key && row.status === 'failed').length;
    if (transient(last) && attempts <= limits.transientRetries) return { action: last.action, reason: `${last.key} failed for a transient reason; try again` };
    return compensate(state, `${last.key} failed for good`);
  }
  switch (state.phase) {
    case 'new': return { action: 'reserve', reason: 'a new order reserves its stock' };
    case 'reserved': return { action: 'charge', reason: 'the stock is reserved, so the customer is charged' };
    case 'charged': return { action: 'ship', reason: 'the payment is done, so the order ships' };
    default: return { action: 'wait', reason: `the order is ${state.phase}` };
  }
}
