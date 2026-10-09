/**
 * The workflow's mechanism: a durable, single-writer order store with explicit recovery. It writes an intent before
 * every remote effect, keeps idempotent remote receipts keyed `order:action`, refuses invalid transitions with a
 * message that names the property that failed, and preserves `pending` when an acknowledgement is lost, so an unknown
 * outcome is reconciled (or, after a look found nothing, sent again under the same key) and never repeated blindly.
 * What to do and what to say are decided by natural language (handle.nl, inform.nl); this file decides only whether a
 * decision is allowed, and carries it out.
 */
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { PluggableSetting } from '@natlang/node';
import type { Action, Decision, Fault, MessageKind, Operation, Outgoing, OutboxEntry, Receipt, WorkflowEvent, WorkflowState } from './types.js';

/** A pluggable part's mode: 'crisp', 'nl' or 'shadow' (runs both and records agreement); 'natlang' and 'natural-language' are deprecated spellings of 'nl', accepted and normalized. */
export type Implementation = Exclude<PluggableSetting, undefined>;

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try { return JSON.parse(await readFile(path, 'utf8')) as T; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return fallback; throw error; }
}
async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, path);
}

const ACTIONS: Action[] = ['reserve', 'charge', 'ship', 'refund', 'release', 'reconcile', 'retry', 'wait'];
const KINDS: MessageKind[] = ['paid', 'shipped', 'delayed', 'problem', 'cancelled'];
const phases: Partial<Record<Action, string>> = { reserve: 'reserved', charge: 'charged', ship: 'shipped', refund: 'refunded', release: 'released' };
/** The phases in which each effect is valid. */
export const prerequisites: Partial<Record<Action, string[]>> = {
  reserve: ['new'], charge: ['reserved', 'charge-failed'], ship: ['charged', 'shipping-failed'],
  refund: ['charged', 'shipping-failed', 'canceling'], release: ['reserved', 'refunded', 'canceling', 'charge-failed'],
};

/**
 * The transition validity check: null when `decision` may be applied to `state` on `event`, otherwise one sentence
 * that names what is wrong and what would be right.
 */
export function validate(state: WorkflowState, event: WorkflowEvent, decision: Decision): string | null {
  if (!event || !['continue', 'cancel', 'reconcile'].includes(event.kind)) return 'event.kind is continue, cancel or reconcile';
  if (!decision || !ACTIONS.includes(decision.action)) return `action is one of ${ACTIONS.join(', ')}`;
  const action = decision.action;
  if (state.pending) {
    if (!['reconcile', 'retry', 'wait'].includes(action))
      return `${state.pending} has an unknown outcome; choose reconcile (look for its receipt), retry (send it again under the same key, after a look found nothing) or wait`;
    if (action === 'retry' && (state.checks ?? 0) < 1) return `${state.pending} has not been looked up yet; choose reconcile first, and retry once a look found no receipt`;
    return null;
  }
  if (event.kind === 'reconcile') {
    if (action === 'wait') return null;
    return 'nothing is pending, so a reconcile event is answered with wait';
  }
  if (action === 'wait') return null;
  if (action === 'reconcile' || action === 'retry') return `nothing is pending, so ${action} has nothing to act on; choose the next effect or wait`;
  if (!prerequisites[action]!.includes(state.phase)) return `${action} is valid in phase ${prerequisites[action]!.join(' or ')}; the order is in phase ${state.phase}`;
  if (event.kind === 'cancel' && !['refund', 'release'].includes(action)) return 'a cancelled order is compensated with refund or release, or waits';
  if (state.history.find(row => row.key === `${state.order_id}:${action}`)?.status === 'done')
    return `${state.order_id}:${action} is already done, and the order is in phase ${state.phase}; choose the step after it`;
  return null;
}

/**
 * The check on a customer message: null when it is acceptable for `state`, otherwise one sentence naming what to
 * change. A kind must be true of the state, and the text uses plain words, without internal operation keys.
 */
export function checkMessage(state: WorkflowState, message: Outgoing): string | null {
  if (!message || !KINDS.includes(message.kind)) return `kind is one of ${KINDS.join(', ')}`;
  if (typeof message.subject !== 'string' || !message.subject.trim() || /\n/.test(message.subject)) return 'the subject is one non-empty line';
  if (typeof message.body !== 'string' || !message.body.trim()) return 'the body has text';
  const leaked = [...state.history.map(row => row.key), ...state.pending ? [state.pending] : []]
    .find(key => message.subject.includes(key) || message.body.includes(key));
  if (leaked) return `the message contains the internal operation key ${leaked}; describe what happened to the order in plain words`;
  const done = (action: Action) => state.history.some(row => row.action === action && row.status === 'done');
  if (message.kind === 'paid' && !done('charge')) return 'a paid message is for an order whose payment is done; choose delayed or problem for this order';
  if (message.kind === 'shipped' && state.phase !== 'shipped') return `a shipped message is for an order in phase shipped; this order is in phase ${state.phase}`;
  if (message.kind === 'cancelled' && !['refunded', 'released'].includes(state.phase)) return `a cancelled message is for an order that was undone (refunded or released); this order is in phase ${state.phase}`;
  return null;
}

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
        return normalize(structuredClone(existing));
      }
      const state: WorkflowState = { order_id: orderId, amount, revision: 0, phase: 'new', pending: '', checks: 0, obligations: [], history: [], outbox: [] };
      workflows[orderId] = state;
      await atomicJson(files.local, workflows);
      this.events.push({ operation: 'workflow.open', order_id: orderId, amount });
      return structuredClone(state);
    });
  }

  async read(orderId: string): Promise<WorkflowState> {
    const state = (await readJson<Record<string, WorkflowState>>((await this.files()).local, {}))[orderId];
    if (!state) throw new Error(`unknown order: ${orderId}`);
    return normalize(structuredClone(state));
  }

  /** The remote's receipt for an operation key, or null. */
  async receipt(key: string): Promise<Receipt | null> {
    return (await readJson<Record<string, Receipt>>((await this.files()).remote, {}))[key] ?? null;
  }

  /** Apply a decision made on revision `expectedRevision`; a changed revision applies nothing and returns the newer state. */
  apply(orderId: string, expectedRevision: number, event: WorkflowEvent, decision: Decision): Promise<WorkflowState> {
    return this.exclusive(async () => {
      const files = await this.files();
      const workflows = await readJson<Record<string, WorkflowState>>(files.local, {});
      const found = workflows[orderId];
      if (!found) throw new Error(`unknown order: ${orderId}`);
      const state = normalize(found);
      workflows[orderId] = state;
      if (state.revision !== expectedRevision) return structuredClone(state);
      const problem = validate(state, event, decision);
      if (problem) throw new Error(problem);
      const action = decision.action;
      if (state.pending) {
        if (action === 'reconcile') {
          const key = state.pending;
          const receipt = (await readJson<Record<string, Receipt>>(files.remote, {}))[key];
          if (receipt) this.ack(state, receipt);
          else { state.checks++; state.obligations = [`outcome unresolved: ${key}`]; }
          this.events.push({ operation: 'workflow.reconcile', order_id: orderId, key, found: !!receipt });
        } else if (action === 'retry') {
          // The same key again: the remote carries out an effect once per key, so sending it again is safe.
          const key = state.pending;
          const entry = state.history.findLast(row => row.key === key)!;
          this.events.push({ operation: 'workflow.retry', order_id: orderId, key });
          await this.perform(files, state, key, entry.action, event.fault);
        }
      } else if (action !== 'wait' && event.kind !== 'reconcile') {
        const key = `${orderId}:${action}`;
        state.pending = key;
        state.obligations = [`outcome unresolved: ${key}`];
        state.history.push({ key, action, status: 'intent', detail: '' });
        state.revision++;
        // The intent is durable before the remote call, so a second process can inspect it.
        await atomicJson(files.local, workflows);
        this.events.push({ operation: 'workflow.intent', order_id: orderId, key, action });
        await this.perform(files, state, key, action, event.fault);
      }
      state.revision++;
      await atomicJson(files.local, workflows);
      return structuredClone(state);
    });
  }

  /** The remote call for an operation whose intent is recorded, with the fault the event injects. */
  private async perform(files: { remote: string }, state: WorkflowState, key: string, action: Action, fault: Fault | undefined): Promise<void> {
    if (fault === 'rate_limit' || fault === 'definite_failure') this.fail(state, action, fault);
    else if (fault === 'lost_request') this.events.push({ operation: 'workflow.request-lost', order_id: state.order_id, key });
    else {
      const remote = await readJson<Record<string, Receipt>>(files.remote, {});
      if (!remote[key]) {
        remote[key] = { key, action, order_id: state.order_id, amount: state.amount, status: 'done' };
        await atomicJson(files.remote, remote);
      }
      if (fault !== 'lost_ack') this.ack(state, remote[key]!);
      else this.events.push({ operation: 'workflow.response-lost', order_id: state.order_id, key });
    }
  }

  /** Queue a customer message once under `key`; a key already queued is left as it is. */
  enqueue(orderId: string, key: string, message: Outgoing): Promise<WorkflowState> {
    return this.exclusive(async () => {
      const files = await this.files();
      const workflows = await readJson<Record<string, WorkflowState>>(files.local, {});
      const found = workflows[orderId];
      if (!found) throw new Error(`unknown order: ${orderId}`);
      const state = normalize(found);
      workflows[orderId] = state;
      if (state.outbox.some(row => row.key === key)) return structuredClone(state);
      const problem = checkMessage(state, message);
      if (problem) throw new Error(problem);
      state.outbox.push({ key, kind: message.kind, subject: message.subject, body: message.body, sent: false });
      state.revision++;
      await atomicJson(files.local, workflows);
      this.events.push({ operation: 'workflow.message', order_id: orderId, key, kind: message.kind });
      return structuredClone(state);
    });
  }

  /** Mark queued messages as delivered. */
  markSent(orderId: string, keys: string[]): Promise<WorkflowState> {
    return this.exclusive(async () => {
      const files = await this.files();
      const workflows = await readJson<Record<string, WorkflowState>>(files.local, {});
      if (!workflows[orderId]) throw new Error(`unknown order: ${orderId}`);
      const state = normalize(workflows[orderId]!);
      workflows[orderId] = state;
      for (const row of state.outbox) if (keys.includes(row.key)) row.sent = true;
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
    state.checks = 0;
    state.obligations = [];
    this.events.push({ operation: 'workflow.ack', order_id: state.order_id, key: receipt.key, phase: state.phase });
  }

  private fail(state: WorkflowState, action: Action, detail: string): void {
    const entry = state.history.findLast(row => row.key === state.pending)!;
    entry.status = 'failed'; entry.detail = detail;
    state.pending = '';
    state.checks = 0;
    state.obligations = [];
    if (action === 'charge') state.phase = 'charge-failed';
    if (action === 'ship') state.phase = 'shipping-failed';
    if (action === 'refund' || action === 'release') state.obligations = [`compensation failed: ${entry.key}`];
    this.events.push({ operation: 'workflow.failure', order_id: state.order_id, key: entry.key, detail });
  }

  async remoteEffects(): Promise<Receipt[]> {
    return Object.values(await readJson<Record<string, Receipt>>((await this.files()).remote, {}));
  }

  /** Add a record to the event log. */
  note(event: Record<string, unknown>): void { this.events.push(event); }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Orders written before `checks` and `outbox` existed read with them empty. */
function normalize(state: WorkflowState): WorkflowState {
  state.checks ??= 0;
  state.outbox ??= [];
  return state;
}

export type { Operation, OutboxEntry };
