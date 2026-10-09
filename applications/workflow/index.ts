/**
 * A durable, single-writer order workflow (inventory, payment, shipping) with explicit recovery, whose policy is
 * natural language. `handle.nl` chooses the next action (and, for an unknown outcome, whether to look again or send the
 * same request again; for a cancellation or failure, which effects to undo), `inform.nl` writes what the customer is
 * told. The service (service.ts) is the mechanism: it validates every transition, writes the intent before the remote
 * effect, keeps idempotent receipts and commits atomically. `step` decides on a snapshot and applies the decision to
 * the revision it read.
 */
import { KeyedEventLoop } from '@natlang/node';
import handle from './handle.nl';
import inform from './inform.nl';
import { ledgerDeclaration, ledgerService } from './ledger.js';
import { WorkflowService, type Implementation } from './service.js';
import type { Decision, Limits, OutboxEntry, Snapshot, WorkflowEvent, WorkflowState } from './types.js';

export * from './service.js';
export * from './ledger.js';
export type * from './types.js';
// The crisp checkers of the refined results (refinements.ts), for a host that runs the policy in its own runtime.
export { refinements } from './refinements.js';

/** What a run of natural-language work needs: the services its stages call. */
export type RunServices = { services: Record<string, unknown>, serviceDeclarations: Record<string, string> };
/** Runs natural-language work in a task with these services (e.g. `(fn, options) => runtime.run(fn, options)`). */
export type Run = <T>(fn: () => Promise<T>, options: RunServices) => Promise<T>;

export const DEFAULT_LIMITS: Limits = { transientRetries: 2, checksBeforeRetry: 2, recheckMs: 30_000 };

export type StepOptions = {
  run: Run,
  /** Which implementation of the next-action hot path runs (default: natural language). */
  policy?: Implementation,
  limits?: Partial<Limits>,
};
export type Stepped = { state: WorkflowState, decision: Decision, message: OutboxEntry | null };

/**
 * Decide and apply one step for an order event from its durable state, then tell the customer if there is something to
 * tell. The decision is made on a snapshot (the order and the remote's receipt for its pending key); `apply` takes it
 * only if the order is still at that revision.
 */
export async function stepFull(service: WorkflowService, orderId: string, event: WorkflowEvent, options: StepOptions): Promise<Stepped> {
  const policy = options.policy ?? 'natural-language';
  const limits = { ...DEFAULT_LIMITS, ...options.limits };
  const run = <T>(fn: () => Promise<T>) => options.run(fn, { services: { ledger: ledgerService() }, serviceDeclarations: { ledger: ledgerDeclaration } });
  const before = await service.read(orderId);
  const snapshot: Snapshot = { state: before, event, receipt: before.pending ? await service.receipt(before.pending) : null };
  // Pluggable hot path: the same Decision from the crisp table or from the natural-language policy.
  const decision = policy === 'crisp' ? handle.crisp(snapshot, limits) : await run(() => handle(snapshot, limits));
  const after = await service.apply(orderId, before.revision, event, decision);
  if (after.revision === before.revision) return { state: after, decision, message: null };
  const outgoing = await run(() => inform({ before, after, decision, event }));
  if (!outgoing) return { state: after, decision, message: null };
  const key = `${orderId}:message:${after.revision}`;
  try {
    const queued = await service.enqueue(orderId, key, outgoing);
    return { state: queued, decision, message: queued.outbox.find(row => row.key === key) ?? null };
  } catch (error) {
    service.note({ operation: 'workflow.message-refused', order_id: orderId, key, detail: String((error as Error).message) });
    return { state: after, decision, message: null };
  }
}

/** Decide and apply one step for an order event, from its current durable state. */
export async function step(service: WorkflowService, orderId: string, event: WorkflowEvent, options: StepOptions): Promise<WorkflowState> {
  return (await stepFull(service, orderId, event, options)).state;
}

/** An event for one order; `wake` comes from the desk itself when an unresolved order is due to be looked at again. */
export type DeskEvent = { id: string; kind: 'continue' | 'cancel' | 'reconcile' | 'wake'; order_id?: string;
  fault?: WorkflowEvent['fault']; at?: number };
export type DeskState = { order: WorkflowState | null; reconcileAt: number | null };

/**
 * Orders handled side by side: one event loop per order, so one order's model decision does not hold up another's,
 * while the service still applies every transition through its single writer. The time an unresolved order is looked at
 * again is state: the policy names it (`Decision.waitMs`, else `reconcileAfterMs`), and it is restored after a restart
 * from the durable `pending`.
 */
export class WorkflowDesk {
  readonly loops: KeyedEventLoop<DeskState, WorkflowState | null, DeskEvent>;

  constructor(readonly service: WorkflowService, options: { run: Run; signal?: never; policy?: Implementation; limits?: Partial<Limits>;
    reconcileAfterMs?: number; onFailure?: (orderId: string, error: unknown) => void;
    /** Runs one event's step with the loop's abort signal (e.g. to cancel natural-language work). */
    guard?: <T>(fn: () => Promise<T>, signal: AbortSignal) => Promise<T> }) {
    const delay = options.reconcileAfterMs ?? 30_000;
    this.loops = new KeyedEventLoop<DeskState, WorkflowState | null, DeskEvent>({
      key: event => event.order_id!,
      initialState: () => ({ order: null, reconcileAt: null }),
      restore: async orderId => {
        const order = await service.read(orderId).catch(() => null);
        return order ? { state: { order, reconcileAt: order.pending ? Date.now() + delay : null }, revision: 0 } : undefined;
      },
      reduce: async (state, event, context) => {
        const orderId = event.order_id ?? state.order?.order_id;
        if (!orderId) throw new Error('an order event needs order_id');
        const kind = event.kind === 'wake' ? (state.order?.pending ? 'reconcile' : 'continue') : event.kind;
        const work = () => stepFull(service, orderId, { kind, ...(event.fault ? { fault: event.fault } : {}) },
          { run: options.run, ...options.policy ? { policy: options.policy } : {}, ...options.limits ? { limits: options.limits } : {} });
        const stepped = await (options.guard ? options.guard(work, context.signal) : work());
        const wait = stepped.decision.waitMs;
        return { order: stepped.state, reconcileAt: stepped.state.pending ? context.now + (wait ?? delay) : wait ? context.now + wait : null };
      },
      view: state => state.order,
      wakeAt: state => state.reconcileAt,
      step: (fn, context) => options.guard ? options.guard(fn, context.signal) : fn(),
      onFailure: (orderId, failure) => options.onFailure?.(orderId, failure.error),
    });
  }

  open(orderId: string, amount: number): Promise<WorkflowState> { return this.service.open(orderId, amount); }

  /** Apply one event to its order; resolves with the order's state after it. */
  async dispatch(event: DeskEvent): Promise<WorkflowState | null> {
    const transition = await this.loops.dispatch(event);
    return transition ? transition.view : null;
  }

  close(): Promise<void> { return this.loops.close(); }
}
