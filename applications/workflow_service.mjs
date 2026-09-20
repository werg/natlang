/** Durable, single-writer fixture for semantic API workflows. */
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

async function readJson(path, fallback) {
  try { return JSON.parse(await readFile(path, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return fallback; throw error; }
}
async function atomicJson(path, value) {
  const temporary = `${path}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, path);
}
const phases = { reserve: 'reserved', charge: 'charged', ship: 'shipped',
  refund: 'refunded', release: 'released' };
const prerequisites = {
  reserve: ['new'], charge: ['reserved'], ship: ['charged'],
  refund: ['charged', 'shipping-failed', 'canceling'],
  release: ['reserved', 'refunded', 'canceling', 'charge-failed'],
};

export class WorkflowService {
  constructor(root) {
    this.root = resolve(root);
    this.events = [];
    this.queue = Promise.resolve();
  }

  async #exclusive(operation) {
    const run = this.queue.then(operation, operation);
    this.queue = run.catch(() => {});
    return run;
  }

  async #files() {
    await mkdir(this.root, { recursive: true });
    return { local: join(this.root, 'workflows.json'),
      remote: join(this.root, 'remote-receipts.json') };
  }

  async open(orderId, amount) {
    return this.#exclusive(async () => {
      if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(orderId) ||
          !Number.isSafeInteger(amount) || amount <= 0) throw new Error('invalid order');
      const files = await this.#files();
      const workflows = await readJson(files.local, {});
      if (workflows[orderId]) {
        if (workflows[orderId].amount !== amount) throw new Error('order amount changed');
        return structuredClone(workflows[orderId]);
      }
      const state = { order_id: orderId, amount, revision: 0, phase: 'new',
        pending: '', obligations: [], history: [] };
      workflows[orderId] = state;
      await atomicJson(files.local, workflows);
      this.events.push({ operation: 'workflow.open', order_id: orderId, amount });
      return structuredClone(state);
    });
  }

  async read(orderId) {
    const files = await this.#files();
    const state = (await readJson(files.local, {}))[orderId];
    if (!state) throw new Error(`unknown order: ${orderId}`);
    return structuredClone(state);
  }

  async apply(orderId, expectedRevision, event, decision) {
    return this.#exclusive(async () => {
      const files = await this.#files();
      const workflows = await readJson(files.local, {});
      const state = workflows[orderId];
      if (!state) throw new Error(`unknown order: ${orderId}`);
      if (state.revision !== expectedRevision) return structuredClone(state);
      if (!event || !['continue', 'cancel', 'reconcile'].includes(event.kind))
        throw new Error('invalid workflow event');
      if (!decision || !['reserve', 'charge', 'ship', 'refund', 'release',
        'reconcile', 'wait'].includes(decision.action)) throw new Error('invalid decision');
      if (event.kind === 'reconcile' || state.pending) {
        if (decision.action !== 'reconcile' && decision.action !== 'wait')
          throw new Error('uncertain operation must be reconciled');
        if (decision.action === 'reconcile') {
          if (!state.pending) throw new Error('nothing to reconcile');
          const remote = await readJson(files.remote, {});
          const receipt = remote[state.pending];
          if (receipt) this.#ack(state, receipt);
          else state.obligations = [`outcome unresolved: ${state.pending}`];
          this.events.push({ operation: 'workflow.reconcile', order_id: orderId,
            key: state.pending || receipt?.key || '', found: !!receipt });
        }
      } else if (decision.action !== 'wait') {
        const action = decision.action;
        if (!prerequisites[action].includes(state.phase))
          throw new Error(`invalid ${action} in ${state.phase}`);
        if (event.kind === 'cancel' && !['refund', 'release'].includes(action))
          throw new Error('cancel requires compensation');
        const key = `${orderId}:${action}`;
        const prior = state.history.find(row => row.key === key);
        if (prior?.status === 'done') throw new Error('completed operation replay');
        state.pending = key;
        state.obligations = [`outcome unresolved: ${key}`];
        state.history.push({ key, action, status: 'intent', detail: '' });
        state.revision++;
        // Intent is durable before a remote call. A second process can inspect it.
        await atomicJson(files.local, workflows);
        this.events.push({ operation: 'workflow.intent', order_id: orderId, key, action });
        const fault = event.fault ?? '';
        if (fault === 'rate_limit' || fault === 'definite_failure') {
          this.#fail(state, action, fault);
        } else {
          const remote = await readJson(files.remote, {});
          if (!remote[key]) {
            remote[key] = { key, action, order_id: orderId, amount: state.amount,
              status: 'done' };
            await atomicJson(files.remote, remote);
          }
          if (fault !== 'lost_ack') this.#ack(state, remote[key]);
          else this.events.push({ operation: 'workflow.response-lost', order_id: orderId, key });
        }
      }
      state.revision++;
      await atomicJson(files.local, workflows);
      return structuredClone(state);
    });
  }

  #ack(state, receipt) {
    if (receipt.order_id !== state.order_id || receipt.key !== state.pending)
      throw new Error('remote receipt does not match intent');
    const entry = state.history.findLast(row => row.key === receipt.key);
    entry.status = 'done';
    state.phase = phases[entry.action];
    state.pending = '';
    state.obligations = [];
    this.events.push({ operation: 'workflow.ack', order_id: state.order_id,
      key: receipt.key, phase: state.phase });
  }

  #fail(state, action, detail) {
    const entry = state.history.findLast(row => row.key === state.pending);
    entry.status = 'failed'; entry.detail = detail;
    state.pending = '';
    state.obligations = [];
    if (action === 'charge') state.phase = 'charge-failed';
    if (action === 'ship') state.phase = 'shipping-failed';
    if (action === 'refund' || action === 'release')
      state.obligations = [`compensation failed: ${entry.key}`];
    this.events.push({ operation: 'workflow.failure', order_id: state.order_id,
      key: entry.key, detail });
  }

  async remoteEffects() {
    const files = await this.#files();
    return Object.values(await readJson(files.remote, {}));
  }

  drainEvents() { return this.events.splice(0); }
}
