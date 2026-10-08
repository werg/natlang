import assert from 'node:assert/strict';
import { test } from 'node:test';
import sqlite3InitModule from '@sqlite.org/sqlite-wasm';
import { createNatlangRuntime, loadVirtualNatlang, caseHashes } from '../dist/index.js';
import { wasmCallStore } from '../dist/calls/store-wasm.js';

// The browser's call store is the shared store over sqlite-wasm (in the browser its database is in OPFS); here the
// same code runs on the package's Node build with an in-memory database.
const sqlite3 = await sqlite3InitModule();
const open = (db = new sqlite3.oo1.DB(':memory:'), scope) => ({ db, ...wasmCallStore('memory', db, { scope }) });

const PRICE = '---\nargs: { id: string }\nreturns: number\n---\nLook up order id and return its total.\n';
const services = () => ({ orders: { lookup: id => ({ id, total: id.length * 10 }) } });
const scripted = calls => { let step = 0; return async () => ({ calls: [calls[step++] ?? ['return_result', { status: 'failed', reason: 'No more calls.' }]] }); };

test('the wasm store records calls with their values kept in the database', async () => {
  const { store } = open();
  const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
  const model = scripted([['eval', { code: 'return orders.lookup(id).total' }], ['return_result', { status: 'success', value: 30 }]]);
  assert.equal(await createNatlangRuntime({ model, calls: store, services: services() }).run(() => price('abc')), 30);
  const [summary] = store.calls();
  const record = store.call(summary.call_id);
  assert.equal(store.value(record.inputs.id), 'abc');
  assert.equal(store.value(record.output), 30);
  assert.deepEqual(store.value(record.effects[0].result), { id: 'abc', total: 30 });
  assert.ok(store.events(summary.call_id)?.length);
  assert.ok(store.diskBytes() > 0);
  store.writeSettings({ auditRate: 0 });
  assert.equal(store.settings().auditRate, 0);
});

test('a compilation in the wasm store serves calls, and rows of an ended session become interrupted', async () => {
  const db = new sqlite3.oo1.DB(':memory:');
  const { store } = open(db, 'session:first');
  const price = loadVirtualNatlang({ 'price.nl': PRICE }, 'price.nl');
  await createNatlangRuntime({ model: scripted([['return_result', { status: 'success', value: 30 }]]), calls: store, services: services(),
    specialization: 'off' }).run(() => price('abc'));
  const record = store.call(store.calls()[0].call_id);
  const text = "import { orders } from 'natlang:services';\nexport const cases = [\n  { when: (args: { id: string }) => args.id.startsWith('a'), run: async (args: { id: string }) => orders.lookup(args.id).total },\n];\n";
  const hashes = caseHashes(text);
  store.saveCompilation({ definitionKey: record.definition.key, definitionId: record.definition.id, definitionName: 'price',
    definitionSource: record.definition.source, interfaceHash: record.definition.interface, programRoot: null,
    files: { 'cases.ts': text }, caseHashes: hashes });
  store.setTier(hashes[0], 'active');
  const runtime = createNatlangRuntime({ model: async () => { throw new Error('the model must not be asked'); }, calls: store, services: services() });
  assert.equal(await runtime.run(() => price('abcd')), 40);
  assert.equal(store.calls()[0].executor, 'crisp');

  store.begin({ callId: 'left/1', parentCallId: null, parentActionIndex: null, taskId: 'left', programId: null, programRoot: null,
    definition: { id: 'x', name: 'x', source: null, key: 'k', interface: 'i', site: 'root' }, modelId: null,
    startedAt: new Date().toISOString(), auditOf: null });
  assert.equal(store.markInterrupted(), 0, 'the session that wrote it is still the one holding the database');
  const next = open(db, 'session:second').store;
  assert.equal(next.calls({ outcome: 'interrupted' }).length, 1);
});
