import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { WorkflowDesk, WorkflowService, checkMessage, stepFull, validate } from '../../applications/dist/workflow/index.js';
import { scriptedModel } from './support/natlang.mjs';

const run = runtime => (fn, options) => runtime.run(fn, options);

// What a model would compute in eval for each stage, as real code over the stage's arguments.
const impl = {
  situation: (state, event) => state.pending ? 'uncertain' : event.kind === 'reconcile' ? 'settled' : event.kind === 'cancel' ? 'cancelling'
    : state.obligations.length || state.phase === 'refunded' ? 'owed' : state.history.at(-1)?.status === 'failed' ? 'failed'
    : ['new', 'reserved', 'charged'].includes(state.phase) ? 'forward' : 'settled',
  advance: state => ({ new: { action: 'reserve', reason: 'reserve' }, reserved: { action: 'charge', reason: 'charge' }, charged: { action: 'ship', reason: 'ship' } }[state.phase]
    ?? { action: 'wait', reason: `the order is ${state.phase}` }),
  failure: operation => operation.detail === 'rate_limit' ? 'transient' : 'definite',
  compensate: state => {
    if (state.history.some(r => r.action === 'ship' && r.status === 'done')) return { steps: [] };
    const d = a => state.history.some(r => r.action === a && r.status === 'done');
    const steps = [];
    if (d('charge') && !d('refund')) steps.push({ action: 'refund', reason: 'the charge is being returned' });
    if (d('reserve') && !d('release')) steps.push({ action: 'release', reason: 'the reserved stock is being returned' });
    return { steps };
  },
  recover: (snapshot, limits) => snapshot.event.kind !== 'reconcile' ? { action: 'wait', reason: 'unknown outcome, looked at again', waitMs: limits.recheckMs }
    : snapshot.receipt ? { action: 'reconcile', reason: 'receipt found' }
    : snapshot.state.checks < limits.checksBeforeRetry ? { action: 'reconcile', reason: 'no receipt yet', waitMs: limits.recheckMs }
    : { action: 'retry', reason: 'no receipt after looking; same key again' },
  moment: report => {
    const d = (s, a) => s.history.some(r => r.action === a && r.status === 'done');
    if (d(report.after, 'charge') && !d(report.before, 'charge')) return 'paid';
    if (report.after.phase === 'shipped' && report.before.phase !== 'shipped') return 'shipped';
    if (report.after.phase === 'released' && report.before.phase !== 'released') return 'cancelled';
    if (report.after.obligations.some(o => o.startsWith('compensation failed')) || report.after.history.at(-1)?.status === 'failed' && report.after.history.at(-1).detail === 'definite_failure') return 'problem';
    if (report.after.pending && !report.before.pending) return 'delayed';
    return 'none';
  },
  facts: (report, kind) => ({ order: report.after.order_id, amount: report.after.amount, summary: `Your order ${kind}.`, next: 'We will keep you posted.' }),
  compose: (kind, told) => ({ kind, subject: `Order ${told.order}: ${kind}`, body: `Hello. ${told.summary} ${told.next}` }),
};
const call = (name, args) => `return (${impl[name]})(${args});`;

const HANDLE = `
  let choice = await choose(snapshot, limits);
  let problem = ledger.validate(snapshot.state, snapshot.event, choice);
  if (problem === null) return choice;
  choice = await choose(snapshot, limits, problem);
  problem = ledger.validate(snapshot.state, snapshot.event, choice);
  if (problem === null) return choice;
  return { action: 'wait', reason: problem, waitMs: limits.recheckMs };`;
const CHOOSE = `
  const where = await situation(snapshot.state, snapshot.event);
  const first = plan => plan.steps.length ? { action: plan.steps[0].action, reason: plan.steps[0].reason } : { action: 'wait', reason: 'nothing is left to undo' };
  if (where === 'uncertain') return await recover(snapshot, limits);
  if (where === 'settled') return { action: 'wait', reason: 'nothing to do' };
  if (where === 'forward') return await advance(snapshot.state);
  if (where === 'cancelling' || where === 'owed') return first(await compensate(snapshot.state));
  const operation = snapshot.state.history.at(-1);
  const kind = await failure(operation);
  const attempts = snapshot.state.history.filter(r => r.key === operation.key && r.status === 'failed').length;
  if (kind === 'transient' && attempts <= limits.transientRetries) return { action: operation.action, reason: 'transient failure, try again' };
  return first(await compensate(snapshot.state));`;
const INFORM = `
  const kind = await moment(report);
  if (kind === 'none') return null;
  const told = await facts(report, kind);
  let draft = await compose(kind, told);
  let problem = ledger.checkMessage(report.after, draft);
  if (problem === null) return draft;
  draft = await compose(kind, told, problem);
  problem = ledger.checkMessage(report.after, draft);
  return problem === null ? draft : null;`;

/** Every stage scripted. `override(stage, opening)` may replace a stage's code. */
function workflowModel({ override = () => undefined, only = null } = {}) {
  const seen = [];
  const model = scriptedModel(async opening => {
    const stage = [['The mechanism applies the decision only if', 'handle'], ['with the stages in your folder. limits are', 'choose'],
      ['Decide where the order in state stands', 'situation'], ['Choose the next forward step', 'advance'], ['is an attempt the remote refused', 'failure'],
      ['List the undos still to do', 'compensate'], ['is the key of an operation that was started', 'recover'],
      ["Write the customer's message with the stages", 'inform'], ['Compare report.before with report.after', 'moment'],
      ['Pick what the customer needs to hear', 'facts'], ['Write the message for the customer from told', 'compose']]
      .find(([marker]) => opening.includes(marker))?.[1];
    seen.push(stage);
    if (only && !only.includes(stage)) return null;
    const replaced = await override(stage, opening);
    if (replaced !== undefined) return replaced;
    switch (stage) {
      case 'handle': return HANDLE;
      case 'choose': return CHOOSE;
      case 'inform': return INFORM;
      case 'situation': return call('situation', 'state, event');
      case 'advance': return call('advance', 'state');
      case 'failure': return call('failure', 'operation');
      case 'compensate': return call('compensate', 'state');
      case 'recover': return call('recover', 'snapshot, limits');
      case 'moment': return call('moment', 'report');
      case 'facts': return call('facts', 'report, kind');
      case 'compose': return call('compose', 'kind, told');
      default: return null;
    }
  });
  return { ...model, seen };
}

async function scenario(events, options = {}) {
  const root = await mkdtemp(join(tmpdir(), 'natlang-workflow-'));
  try {
    const service = new WorkflowService(root);
    await service.open('order1', 1200);
    const model = workflowModel(options.model);
    const runtime = createNatlangRuntime({ model: model.driver });
    const decisions = [], states = [];
    for (const event of events) {
      const stepped = await stepFull(service, 'order1', event, { run: run(runtime), policy: options.policy, limits: options.limits });
      decisions.push(stepped.decision.action); states.push(stepped.state);
    }
    return { service, decisions, states, state: states.at(-1), model, effects: (await service.remoteEffects()).map(row => row.action) };
  } finally { await rm(root, { recursive: true, force: true }); }
}

test('natlang recovers a lost payment acknowledgement without a second charge, and tells the customer', async () => {
  const result = await scenario([{ kind: 'continue' }, { kind: 'continue', fault: 'lost_ack' }, { kind: 'reconcile' }, { kind: 'continue' }]);
  assert.deepEqual(result.decisions, ['reserve', 'charge', 'reconcile', 'ship']);
  assert.equal(result.state.phase, 'shipped');
  assert.equal(result.state.pending, '');
  assert.deepEqual(result.effects, ['reserve', 'charge', 'ship']);
  assert.deepEqual(result.state.outbox.map(row => row.kind), ['delayed', 'paid', 'shipped']);
  assert.ok(result.state.outbox.every(row => !row.body.includes('order1:')));
});

test('an unknown outcome with no receipt is looked up, then sent again under the same key', async () => {
  const events = [{ kind: 'continue' }, { kind: 'continue', fault: 'lost_request' }, { kind: 'reconcile' }, { kind: 'reconcile' }, { kind: 'reconcile' }];
  const result = await scenario(events);
  assert.deepEqual(result.decisions, ['reserve', 'charge', 'reconcile', 'reconcile', 'retry']);
  assert.deepEqual(result.states.map(state => state.checks), [0, 0, 1, 2, 0]);
  assert.equal(result.state.phase, 'charged');
  assert.deepEqual(result.effects, ['reserve', 'charge']);
});

test('a transient failure is retried, a definite one is compensated in order, and a failed compensation is retried', async () => {
  const events = [{ kind: 'continue' }, { kind: 'continue', fault: 'rate_limit' }, { kind: 'continue' }, { kind: 'continue', fault: 'definite_failure' },
    { kind: 'continue' }, { kind: 'continue' }];
  const result = await scenario(events);
  assert.deepEqual(result.decisions, ['reserve', 'charge', 'charge', 'ship', 'refund', 'release']);
  assert.equal(result.state.phase, 'released');
  assert.deepEqual(result.effects, ['reserve', 'charge', 'refund', 'release']);
  assert.deepEqual(result.state.outbox.map(row => row.kind), ['paid', 'problem', 'cancelled']);
  const owed = await scenario([{ kind: 'continue' }, { kind: 'continue' }, { kind: 'cancel', fault: 'rate_limit' }, { kind: 'continue' }, { kind: 'continue' }]);
  assert.deepEqual(owed.decisions, ['reserve', 'charge', 'refund', 'refund', 'release']);
  assert.match(owed.states[2].obligations[0], /compensation failed/);
  assert.equal(owed.state.phase, 'released');
});

test('the crisp policy makes the same decisions as the natural-language one, without a model call for them', async () => {
  const events = [{ kind: 'continue' }, { kind: 'continue', fault: 'lost_request' }, { kind: 'reconcile' }, { kind: 'reconcile' }, { kind: 'reconcile' },
    { kind: 'continue', fault: 'rate_limit' }, { kind: 'continue' }, { kind: 'cancel' }, { kind: 'continue' }];
  const natural = await scenario(events);
  const crisp = await scenario(events, { policy: 'crisp', model: { only: ['inform', 'moment', 'facts', 'compose'] } });
  assert.deepEqual(crisp.decisions, natural.decisions);
  assert.deepEqual(crisp.states.map(state => [state.phase, state.pending, state.checks]), natural.states.map(state => [state.phase, state.pending, state.checks]));
  assert.ok(!crisp.model.seen.includes('handle') && !crisp.model.seen.includes('choose'));
  assert.ok(natural.model.seen.includes('choose'));
});

test('a refused choice goes back with the problem; a choice still refused waits', async () => {
  const events = [{ kind: 'continue' }, { kind: 'continue', fault: 'lost_ack' }, { kind: 'continue' }];
  let asked = 0;
  const first = await scenario(events, { model: { override: stage => {
    if (stage !== 'choose') return undefined;
    asked++;
    // The first choice for an order with a pending charge repeats the charge; with the problem it waits.
    return `if (snapshot.state.pending) return problem ? { action: 'wait', reason: 'looked at later' } : { action: 'charge', reason: 'again' };\n${CHOOSE}`;
  } } });
  assert.deepEqual(first.decisions, ['reserve', 'charge', 'wait']);
  assert.deepEqual(first.effects, ['reserve', 'charge']);
  const stubborn = await scenario(events, { model: { override: stage =>
    stage === 'choose' ? `if (snapshot.state.pending) return { action: 'ship', reason: 'impatient' };\n${CHOOSE}` : undefined } });
  assert.deepEqual(stubborn.decisions.slice(2), ['wait']);
  assert.ok(asked >= 2);
});

test('the mechanism names the property a decision lacks, and keeps its guarantees across a restart', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-workflow-'));
  try {
    let service = new WorkflowService(root);
    let state = await service.open('order2', 700);
    state = await service.apply('order2', state.revision, { kind: 'continue' }, { action: 'reserve', reason: '' });
    state = await service.apply('order2', state.revision, { kind: 'continue', fault: 'lost_ack' }, { action: 'charge', reason: '' });
    assert.equal(state.pending, 'order2:charge');
    service = new WorkflowService(root);
    assert.equal((await service.read('order2')).pending, 'order2:charge');
    await assert.rejects(service.apply('order2', state.revision, { kind: 'continue' }, { action: 'charge', reason: '' }), /unknown outcome; choose reconcile/);
    await assert.rejects(service.apply('order2', state.revision, { kind: 'continue' }, { action: 'retry', reason: '' }), /not been looked up yet/);
    const stale = await service.apply('order2', 0, { kind: 'continue' }, { action: 'ship', reason: 'stale' });
    assert.equal(stale.revision, state.revision, 'a stale revision applies nothing');
    state = await service.apply('order2', state.revision, { kind: 'reconcile' }, { action: 'reconcile', reason: '' });
    assert.equal(state.phase, 'charged');
    assert.equal(validate(state, { kind: 'continue' }, { action: 'charge', reason: '' }), 'charge is valid in phase reserved or charge-failed; the order is in phase charged');
    assert.match(validate(state, { kind: 'continue' }, { action: 'retry', reason: '' }), /nothing is pending/);
    assert.match(validate(state, { kind: 'cancel' }, { action: 'ship', reason: '' }), /compensated with refund or release/);
    state = await service.apply('order2', state.revision, { kind: 'continue', fault: 'definite_failure' }, { action: 'ship', reason: '' });
    assert.equal(state.phase, 'shipping-failed');
    state = await service.apply('order2', state.revision, { kind: 'continue', fault: 'rate_limit' }, { action: 'refund', reason: '' });
    assert.match(state.obligations[0], /compensation failed/);
    assert.equal((await service.remoteEffects()).filter(row => row.action === 'charge').length, 1);
    state = await service.apply('order2', state.revision, { kind: 'continue' }, { action: 'refund', reason: '' });
    assert.equal(state.phase, 'refunded');
    await assert.rejects(service.apply('order2', state.revision, { kind: 'continue' }, { action: 'refund', reason: '' }), /refund is valid in phase/);
    // Messages are checked against the state and for internal keys, and queued once per key.
    assert.match(checkMessage(state, { kind: 'shipped', subject: 'Shipped', body: 'On its way.' }), /order is in phase refunded/);
    assert.match(checkMessage(state, { kind: 'cancelled', subject: 'Cancelled', body: 'See order2:refund.' }), /internal operation key order2:refund/);
    const ok = { kind: 'cancelled', subject: 'Order order2 cancelled', body: 'Your payment is returned.' };
    state = await service.enqueue('order2', 'order2:message:1', ok);
    const again = await service.enqueue('order2', 'order2:message:1', ok);
    assert.equal(again.outbox.length, 1);
    assert.equal((await service.markSent('order2', ['order2:message:1'])).outbox[0].sent, true);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('the desk decides orders side by side and looks at an unacknowledged charge again on its own', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-workflow-'));
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  // The decision for order "slow" waits until released; order "quick" must not wait for it.
  const model = workflowModel({ override: async (stage, opening) => {
    if (stage === 'handle' && opening.includes('"order_id": "slow"')) await gate;
    return undefined;
  } });
  const runtime = createNatlangRuntime({ model: model.driver });
  const failures = [];
  const desk = new WorkflowDesk(new WorkflowService(root), { run: run(runtime), reconcileAfterMs: 50, limits: { recheckMs: 50 },
    onFailure: (order, error) => failures.push([order, String(error)]) });
  try {
    await desk.open('slow', 300); await desk.open('quick', 500);
    const slow = desk.dispatch({ id: 's1', kind: 'continue', order_id: 'slow' });
    assert.equal((await desk.dispatch({ id: 'q1', kind: 'continue', order_id: 'quick' })).phase, 'reserved');
    const uncertain = await desk.dispatch({ id: 'q2', kind: 'continue', order_id: 'quick', fault: 'lost_ack' });
    assert.equal(uncertain.pending, 'quick:charge');
    release();
    assert.equal((await slow).phase, 'reserved');
    await new Promise(resolve => setTimeout(resolve, 600));
    const quick = await desk.service.read('quick');
    assert.equal(quick.phase, 'charged'); assert.equal(quick.pending, '');
    assert.equal((await desk.service.remoteEffects()).filter(row => row.action === 'charge').length, 1);
    assert.deepEqual(failures, []);
  } finally { await desk.close(); await rm(root, { recursive: true, force: true }); }
});
