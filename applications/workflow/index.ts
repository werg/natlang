/**
 * A durable, single-writer order workflow (inventory, payment, shipping) with explicit recovery.
 * Natlang chooses the next action from the durable state; the service rejects invalid transitions,
 * writes an intent before each external effect, keeps an idempotent remote receipt, and preserves
 * `pending` when an acknowledgement is lost, so an unknown outcome is reconciled rather than retried.
 */
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { nl } from '@natlang/node';

export type Action = 'reserve' | 'charge' | 'ship' | 'refund' | 'release' | 'reconcile' | 'wait';
export type WorkflowEvent = { kind: 'continue' | 'cancel' | 'reconcile', fault?: 'rate_limit' | 'definite_failure' | 'lost_ack' };
export type Operation = { key: string, action: Action, status: 'intent' | 'done' | 'failed', detail: string };
export type WorkflowState = { order_id: string, amount: number, revision: number, phase: string, pending: string,
  obligations: string[], history: Operation[] };
export type Decision = { action: Action, reason: string };
type Receipt = { key: string, action: Action, order_id: string, amount: number, status: 'done' };

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try { return JSON.parse(await readFile(path, 'utf8')) as T; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return fallback; throw error; }
}
async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, path);
}
const phases: Partial<Record<Action, string>> = { reserve: 'reserved', charge: 'charged', ship: 'shipped', refund: 'refunded', release: 'released' };
const prerequisites: Partial<Record<Action, string[]>> = {
  reserve: ['new'], charge: ['reserved'], ship: ['charged'],
  refund: ['charged', 'shipping-failed', 'canceling'], release: ['reserved', 'refunded', 'canceling', 'charge-failed'],
};

export class WorkflowService {
  private readonly root: string;
  private readonly events: Record<string, unknown>[] = [];
  private queue: Promise<unknown> = Promise.resolve();

  constructor(root: string) { this.root = resolve(root); }

  private exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.queue.then(operation, operation);
    this.queue = run.catch(() => {});
    return run;
  }

  private async files(): Promise<{ local: string, remote: string }> {
    await mkdir(this.root, { recursive: true });
    return { local: join(this.root, 'workflows.json'), remote: join(this.root, 'remote-receipts.json') };
  }

  open(orderId: string, amount: number): Promise<WorkflowState> {
    return this.exclusive(async () => {
      if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(orderId) || !Number.isSafeInteger(amount) || amount <= 0) throw new Error('invalid order');
      const files = await this.files();
      const workflows = await readJson<Record<string, WorkflowState>>(files.local, {});
      const existing = workflows[orderId];
      if (existing) {
        if (existing.amount !== amount) throw new Error('order amount changed');
        return structuredClone(existing);
      }
      const state: WorkflowState = { order_id: orderId, amount, revision: 0, phase: 'new', pending: '', obligations: [], history: [] };
      workflows[orderId] = state;
      await atomicJson(files.local, workflows);
      this.events.push({ operation: 'workflow.open', order_id: orderId, amount });
      return structuredClone(state);
    });
  }

  async read(orderId: string): Promise<WorkflowState> {
    const state = (await readJson<Record<string, WorkflowState>>((await this.files()).local, {}))[orderId];
    if (!state) throw new Error(`unknown order: ${orderId}`);
    return structuredClone(state);
  }

  apply(orderId: string, expectedRevision: number, event: WorkflowEvent, decision: Decision): Promise<WorkflowState> {
    return this.exclusive(async () => {
      const files = await this.files();
      const workflows = await readJson<Record<string, WorkflowState>>(files.local, {});
      const state = workflows[orderId];
      if (!state) throw new Error(`unknown order: ${orderId}`);
      if (state.revision !== expectedRevision) return structuredClone(state);
      if (!event || !['continue', 'cancel', 'reconcile'].includes(event.kind)) throw new Error('invalid workflow event');
      if (!decision || !['reserve', 'charge', 'ship', 'refund', 'release', 'reconcile', 'wait'].includes(decision.action))
        throw new Error('invalid decision');
      if (event.kind === 'reconcile' || state.pending) {
        if (decision.action !== 'reconcile' && decision.action !== 'wait') throw new Error('uncertain operation must be reconciled');
        if (decision.action === 'reconcile') {
          if (!state.pending) throw new Error('nothing to reconcile');
          const receipt = (await readJson<Record<string, Receipt>>(files.remote, {}))[state.pending];
          const key = state.pending;
          if (receipt) this.ack(state, receipt);
          else state.obligations = [`outcome unresolved: ${state.pending}`];
          this.events.push({ operation: 'workflow.reconcile', order_id: orderId, key, found: !!receipt });
        }
      } else if (decision.action !== 'wait') {
        const action = decision.action;
        if (!prerequisites[action]?.includes(state.phase)) throw new Error(`invalid ${action} in ${state.phase}`);
        if (event.kind === 'cancel' && !['refund', 'release'].includes(action)) throw new Error('cancel requires compensation');
        const key = `${orderId}:${action}`;
        if (state.history.find(row => row.key === key)?.status === 'done') throw new Error('completed operation replay');
        state.pending = key;
        state.obligations = [`outcome unresolved: ${key}`];
        state.history.push({ key, action, status: 'intent', detail: '' });
        state.revision++;
        // The intent is durable before the remote call, so a second process can inspect it.
        await atomicJson(files.local, workflows);
        this.events.push({ operation: 'workflow.intent', order_id: orderId, key, action });
        const fault = event.fault ?? '';
        if (fault === 'rate_limit' || fault === 'definite_failure') this.fail(state, action, fault);
        else {
          const remote = await readJson<Record<string, Receipt>>(files.remote, {});
          if (!remote[key]) {
            remote[key] = { key, action, order_id: orderId, amount: state.amount, status: 'done' };
            await atomicJson(files.remote, remote);
          }
          if (fault !== 'lost_ack') this.ack(state, remote[key]);
          else this.events.push({ operation: 'workflow.response-lost', order_id: orderId, key });
        }
      }
      state.revision++;
      await atomicJson(files.local, workflows);
      return structuredClone(state);
    });
  }

  private ack(state: WorkflowState, receipt: Receipt): void {
    if (receipt.order_id !== state.order_id || receipt.key !== state.pending) throw new Error('remote receipt does not match intent');
    const entry = state.history.findLast(row => row.key === receipt.key)!;
    entry.status = 'done';
    state.phase = phases[entry.action]!;
    state.pending = '';
    state.obligations = [];
    this.events.push({ operation: 'workflow.ack', order_id: state.order_id, key: receipt.key, phase: state.phase });
  }

  private fail(state: WorkflowState, action: Action, detail: string): void {
    const entry = state.history.findLast(row => row.key === state.pending)!;
    entry.status = 'failed'; entry.detail = detail;
    state.pending = '';
    state.obligations = [];
    if (action === 'charge') state.phase = 'charge-failed';
    if (action === 'ship') state.phase = 'shipping-failed';
    if (action === 'refund' || action === 'release') state.obligations = [`compensation failed: ${entry.key}`];
    this.events.push({ operation: 'workflow.failure', order_id: state.order_id, key: entry.key, detail });
  }

  async remoteEffects(): Promise<Receipt[]> {
    return Object.values(await readJson<Record<string, Receipt>>((await this.files()).remote, {}));
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Decide and apply one step for an order event, from its current durable state. */
export async function step(service: WorkflowService, orderId: string, event: WorkflowEvent): Promise<WorkflowState> {
  const current = await service.read(orderId);
  const decision: Decision = await nl`Choose exactly one next action for the order in current, given event: reserve, charge, ship,
refund, release, reconcile or wait. When current.pending is nonempty, choose reconcile for a reconciliation event and otherwise
wait; never repeat an uncertain charge. For a new order reserve inventory; after reserve charge; after charge ship. On a definite
shipping failure or a cancellation after payment, refund, then release inventory. On a charge failure, release inventory. A
completed shipment waits. Use the history and obligations, and give a brief reason.`(current, event);
  return service.apply(orderId, current.revision, event, decision);
}
