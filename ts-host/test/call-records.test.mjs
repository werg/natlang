import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, CallStore, caseHashes } from '../dist/index.js';

const freshStore = () => CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-calls-')));
const done = store => { const root = store.root; store.close(); rmSync(root, { recursive: true, force: true }); };

/** A model that plays `calls` in order and records every request it was sent. */
function scripted(calls) {
  const requests = [];
  let step = 0;
  const model = async request => {
    requests.push(request);
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'script ended' }]] };
  };
  return { model, requests };
}

const PRICE = '---\nargs: { id: string }\nreturns: number\n---\nLook up order id and return its total.\n';
const orders = () => {
  const log = [];
  return { log, orders: { lookup(id) { log.push(id); return { id, total: id.length * 10 }; }, async refund(id) { log.push(`refund ${id}`); return { ok: true }; } } };
};

test('a call record keeps exact inputs, output, effects, eval programs and cost', async () => {
  const store = freshStore();
  try {
    const { model } = scripted([['eval', { code: 'const order = orders.lookup(id);\nreturn order.total' }],
      ['return_result', { status: 'success', value: 30 }]]);
    const service = orders();
    const runtime = createNatlangRuntime({ model, calls: store, services: { orders: service.orders }, programRoot: '/tmp/program' });
    const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
    assert.equal(await runtime.run(() => price('abc')), 30);
    const [summary] = store.calls();
    const record = store.call(summary.call_id);
    assert.equal(record.version, 'natlang.calls/1');
    assert.equal(record.executor.kind, 'agent');
    assert.equal(record.outcome, 'done');
    assert.equal(record.program_root, '/tmp/program');
    assert.deepEqual(store.value(record.inputs.id), 'abc');
    assert.equal(store.value(record.output), 30);
    assert.equal(record.effects.length, 1);
    assert.equal(record.effects[0].service, 'orders');
    assert.deepEqual(store.value(record.effects[0].args), ['abc']);
    assert.deepEqual(store.value(record.effects[0].result), { id: 'abc', total: 30 });
    assert.equal(record.approach.evals.length, 1);
    assert.match(record.approach.hash, /^[0-9a-f]{24}$/);
    assert.equal(record.cost.evals, 1);
    assert.equal(record.features['str:id'], 'abc');
    assert.equal(record.features['tok:id:abc'], true);
    assert.ok(store.events(summary.call_id).some(event => event.kind === 'action'));
    assert.equal(store.hot()[0].calls, 1);
  } finally { done(store); }
});

test('the same approach on different inputs normalizes to the same hash', async () => {
  const store = freshStore();
  try {
    const service = orders();
    const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
    for (const id of ['abc', 'wxyz']) {
      const { model } = scripted([['eval', { code: `const order = orders.lookup(${JSON.stringify(id)});\nreturn order.total` }],
        ['return_result', { status: 'success', value: id.length * 10 }]]);
      await createNatlangRuntime({ model, calls: store, services: { orders: service.orders } }).run(() => price(id));
    }
    const hashes = store.calls().map(call => call.approach_hash);
    assert.equal(hashes.length, 2);
    assert.equal(hashes[0], hashes[1]);
  } finally { done(store); }
});

test('a child call names its parent and the eval that started it', async () => {
  const store = freshStore();
  try {
    const files = { 'root.nl': '---\nargs: { id: string }\nreturns: number\n---\nAsk price for id.\n', 'root/price.nl': PRICE };
    let step = 0;
    const model = async request => {
      const opening = String(request.messages.find(message => message.role === 'user')?.content ?? '');
      if (opening.includes('Look up order')) return { calls: [['return_result', { status: 'success', value: 7 }]] };
      return { calls: [step++ === 0 ? ['eval', { code: 'return await price(id)' }] : ['return_result', { status: 'success', value: 7 }]] };
    };
    const root = loadVirtualNatlang(files, 'root.nl');
    await createNatlangRuntime({ model, calls: store, services: orders() }).run(() => root('q'));
    const calls = store.calls();
    const child = calls.find(call => call.definition_name === 'price');
    const parent = calls.find(call => call.definition_name === 'root');
    assert.equal(child.parent_call_id, parent.call_id);
    const record = store.call(child.call_id);
    assert.equal(typeof record.parent_action_index, 'number');
    assert.equal(store.children(parent.call_id).length, 1);
  } finally { done(store); }
});

/** Store `cases` as the current compilation of price.nl (learned from one recorded call) with the given tiers. */
async function compile(store, casesText, tiers) {
  const service = orders();
  const { model } = scripted([['return_result', { status: 'success', value: 30 }]]);
  const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
  await createNatlangRuntime({ model, calls: store, services: { orders: service.orders }, specialization: 'off' }).run(() => price('abc'));
  const [seed] = store.calls();
  const record = store.call(seed.call_id);
  const hashes = caseHashes(casesText);
  store.saveCompilation({ definitionKey: record.definition.key, definitionId: record.definition.id, definitionName: 'price',
    definitionSource: record.definition.source, interfaceHash: record.definition.interface, programRoot: null,
    files: { 'cases.ts': casesText }, caseHashes: hashes, links: [{ caseHash: hashes[0], callId: seed.call_id, role: 'training' }] });
  hashes.forEach((hash, index) => store.setTier(hash, tiers[index] ?? 'active'));
  return { price, hashes };
}

const LOOKUP_CASES = `import { orders } from 'natlang:services';
export const cases = [
  { when: (args: { id: string }) => args.id.startsWith('a'),
    run: async (args: { id: string }) => orders.lookup(args.id).total },
];
`;

test('an active case serves the call without the model, and the record says which case', async () => {
  const store = freshStore();
  try {
    const { price, hashes } = await compile(store, LOOKUP_CASES, ['active']);
    const service = orders();
    const runtime = createNatlangRuntime({ model: async () => { throw new Error('the model must not be asked'); }, calls: store,
      services: { orders: service.orders } });
    assert.equal(await runtime.run(() => price('abcd')), 40);
    assert.deepEqual(service.log, ['abcd']);
    const [served] = store.calls({ executor: 'crisp' });
    const record = store.call(served.call_id);
    assert.equal(record.executor.case_hash, hashes[0]);
    assert.equal(record.effects[0].by, 'crisp');
    assert.deepEqual(store.value(record.effects[0].result), { id: 'abcd', total: 40 });
    assert.equal(store.caseStats(hashes[0]).served, 1);
    assert.equal(store.caseCalls(hashes[0], 'served')[0].call_id, served.call_id);
  } finally { done(store); }
});

test('a guard that does not admit leaves the call to the agent', async () => {
  const store = freshStore();
  try {
    const { price } = await compile(store, LOOKUP_CASES, ['active']);
    const { model, requests } = scripted([['return_result', { status: 'success', value: 5 }]]);
    assert.equal(await createNatlangRuntime({ model, calls: store, services: orders() }).run(() => price('zz')), 5);
    assert.equal(requests.length, 1);
  } finally { done(store); }
});

test('a case that fails after an effect hands the call to the agent with what already happened', async () => {
  const store = freshStore();
  try {
    const failing = `import { orders } from 'natlang:services';
export const cases = [
  { when: (args: { id: string }) => true,
    run: async (args: { id: string }) => { await orders.refund(args.id); throw new Error('no total for refunded orders'); } },
];
`;
    const { price, hashes } = await compile(store, failing, ['active']);
    const { model, requests } = scripted([['return_result', { status: 'success', value: 0 }]]);
    const service = orders();
    assert.equal(await createNatlangRuntime({ model, calls: store, services: { orders: service.orders } }).run(() => price('abc')), 0);
    assert.deepEqual(service.log, ['refund abc']);
    const opening = String(requests[0].messages.find(message => message.role === 'user').content);
    assert.match(opening, /compiled fast path for this call started and stopped: no total for refunded orders/);
    assert.match(opening, /orders\.refund\("abc"\) returned \{"ok":true\}/);
    const [record] = store.calls({ executor: 'crisp-agent' }).map(call => store.call(call.call_id));
    assert.equal(record.executor.case_hash, hashes[0]);
    assert.match(record.executor.case_error, /no total/);
    assert.equal(store.caseStats(hashes[0]).handed_off, 1);
  } finally { done(store); }
});

test('Deopt before any effect runs the agent as usual, with no note', async () => {
  const store = freshStore();
  try {
    const declining = `import { Deopt } from '@natlang/node';
export const cases = [
  { when: (args: { id: string }) => true, run: async (args: { id: string }) => { throw new Deopt('not this one'); } },
];
`;
    const { price, hashes } = await compile(store, declining, ['active']);
    const { model, requests } = scripted([['return_result', { status: 'success', value: 1 }]]);
    assert.equal(await createNatlangRuntime({ model, calls: store, services: orders() }).run(() => price('abc')), 1);
    assert.doesNotMatch(String(requests[0].messages.find(message => message.role === 'user').content), /fast path/);
    assert.equal(store.caseStats(hashes[0]).handed_off, 0);
  } finally { done(store); }
});

test('a case whose value has the wrong type hands the call to the agent', async () => {
  const store = freshStore();
  try {
    const wrong = `export const cases = [{ when: (args: { id: string }) => true, run: async (args: { id: string }) => 'not a number' }];\n`;
    const { price } = await compile(store, wrong, ['active']);
    const { model } = scripted([['return_result', { status: 'success', value: 2 }]]);
    assert.equal(await createNatlangRuntime({ model, calls: store, services: orders() }).run(() => price('abc')), 2);
    assert.equal(store.calls({ executor: 'crisp-agent' }).length, 1);
  } finally { done(store); }
});

test('a shadow case never serves; the agent call is queued for a shadow replay', async () => {
  const store = freshStore();
  try {
    const { price, hashes } = await compile(store, LOOKUP_CASES, ['shadow']);
    const { model } = scripted([['return_result', { status: 'success', value: 40 }]]);
    assert.equal(await createNatlangRuntime({ model, calls: store, services: orders() }).run(() => price('abcd')), 40);
    const jobs = store.pendingJobs();
    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].kind, 'shadow');
    assert.equal(jobs[0].case_hash, hashes[0]);
  } finally { done(store); }
});

test('specialization off in the machine settings keeps compilations from serving', async () => {
  const store = freshStore();
  try {
    const { price } = await compile(store, LOOKUP_CASES, ['active']);
    store.writeSettings({ specialization: 'off' });
    const { model, requests } = scripted([['return_result', { status: 'success', value: 3 }]]);
    assert.equal(await createNatlangRuntime({ model, calls: store, services: orders() }).run(() => price('abc')), 3);
    assert.equal(requests.length, 1);
  } finally { done(store); }
});

test('excluded arguments are recorded by type only', async () => {
  const store = freshStore();
  try {
    const { model } = scripted([['return_result', { status: 'success', value: 1 }]]);
    const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
    await createNatlangRuntime({ model, calls: store, recording: { exclude: ['price.id'] } }).run(() => price('secret'));
    const record = store.call(store.calls()[0].call_id);
    assert.deepEqual(record.inputs.id, { complete: false, reason: 'excluded', type: 'string' });
    assert.equal(Object.keys(record.features).length, 0);
  } finally { done(store); }
});

test('eviction drops unpinned calls first and keeps calls cited by a case', async () => {
  const store = freshStore();
  try {
    const { hashes } = await compile(store, LOOKUP_CASES, ['active']);
    const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
    for (let index = 0; index < 5; index++) {
      const { model } = scripted([['return_result', { status: 'success', value: index }]]);
      await createNatlangRuntime({ model, calls: store, services: orders(), specialization: 'off' }).run(() => price(`z${'x'.repeat(index)}`));
    }
    const pinned = store.caseCalls(hashes[0], 'training')[0].call_id;
    assert.equal(store.calls({ limit: 100 }).length, 6);
    store.evict(1);
    assert.deepEqual(store.calls({ limit: 100 }).map(call => call.call_id), [pinned]);
    assert.ok(store.call(pinned));
  } finally { done(store); }
});
