import assert from 'node:assert/strict';
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { BudgetExhausted, UsageGateway } from '../dist/evaluation/usage.js';
import { RunStore } from '../dist/optimization/run-store.js';
import { ENGINE_VERSION, readCheckpoint, saveCheckpoint } from '../dist/optimization/checkpoint.js';

const limits = overrides => ({ maxRollouts: 4, maxProposals: 3, maxModelCalls: 4, ...overrides });
const request = { messages: [], tools: [], seed: null, max_tokens: null };
function temp() { return mkdtempSync(join(tmpdir(), 'natlang-adaptation-budget-')); }

test('usage totals accumulate actual provider tokens and preserve unavailable token totals as unknown', async () => {
  const gateway = new UsageGateway(limits({ maxInputTokens: 10, maxOutputTokens: 10 }));
  await gateway.request(async () => ({ text: 'one', prompt_tokens: 3, completion_tokens: 2 }), request);
  assert.equal(gateway.ledger.usage.modelCalls, 1);
  assert.equal(gateway.ledger.usage.inputTokens, 3);
  assert.equal(gateway.ledger.usage.outputTokens, 2);
  await gateway.request(async () => ({ text: 'two' }), request);
  assert.equal(gateway.ledger.usage.modelCalls, 2);
  assert.equal(gateway.ledger.usage.inputTokens, null);
  assert.equal(gateway.ledger.usage.outputTokens, null);
  assert.throws(() => gateway.check(), error => error instanceof BudgetExhausted && error.dimension === 'inputTokens');
});

test('rollout, proposal, and model request reservations stop exactly at hard limits', async () => {
  const gateway = new UsageGateway(limits({ maxRollouts: 1, maxProposals: 2, maxModelCalls: 1 }));
  gateway.reserve('rollouts');
  assert.throws(() => gateway.reserve('rollouts'), error => error instanceof BudgetExhausted && error.dimension === 'rollouts');
  assert.equal(gateway.ledger.rollouts, 1);
  gateway.reserve('proposals', 2);
  assert.throws(() => gateway.reserve('proposals'), error => error instanceof BudgetExhausted && error.dimension === 'proposals');
  assert.equal(gateway.ledger.proposals, 2);
  await gateway.request(async () => ({ text: 'ok', prompt_tokens: 1, completion_tokens: 1 }), request);
  await assert.rejects(gateway.request(async () => ({ text: 'over budget' }), request),
    error => error instanceof BudgetExhausted && error.dimension === 'modelCalls');
  assert.equal(gateway.ledger.usage.modelCalls, 1);
});

test('cancellation and model failures are accounted as requests with unknown token use', async () => {
  const cancelled = new UsageGateway(limits());
  const controller = new AbortController();
  let forwarded;
  const pending = cancelled.request((_request, signal) => { forwarded = signal; return new Promise(() => {}); }, request, controller.signal);
  await Promise.resolve();
  assert.equal(forwarded, controller.signal, 'drivers receive the coordinator cancellation signal');
  controller.abort(new Error('stop now'));
  await assert.rejects(pending, /stop now/);
  assert.equal(cancelled.ledger.usage.modelCalls, 1);
  assert.equal(cancelled.ledger.unknownRequests, 1);
  assert.equal(cancelled.ledger.usage.inputTokens, null);
  assert.equal(cancelled.ledger.usage.outputTokens, null);

  const failed = new UsageGateway(limits());
  await assert.rejects(failed.request(async () => { throw new Error('provider down'); }, request), /provider down/);
  assert.equal(failed.ledger.usage.modelCalls, 1);
  assert.equal(failed.ledger.unknownRequests, 1);
  assert.equal(failed.ledger.usage.inputTokens, null);
  assert.equal(failed.ledger.usage.outputTokens, null);

  const synchronousAbort = new UsageGateway(limits());
  const duringCall = new AbortController();
  const alsoPending = synchronousAbort.request(() => {
    duringCall.abort(new Error('cancelled while opening request'));
    return new Promise(() => {});
  }, request, duringCall.signal);
  await assert.rejects(alsoPending, /cancelled while opening request/);
  assert.equal(synchronousAbort.ledger.unknownRequests, 1);
});

test('reported tokens beyond a hard cap are recorded as unavoidable overrun and block later work', async () => {
  const gateway = new UsageGateway(limits({ maxInputTokens: 5, maxOutputTokens: 4 }));
  await gateway.request(async () => ({ text: 'large', prompt_tokens: 6, completion_tokens: 5 }), request);
  const ledger = gateway.snapshot();
  assert.equal(ledger.usage.inputTokens, 6);
  assert.equal(ledger.usage.outputTokens, 5);
  assert.deepEqual(new Set(ledger.overrun), new Set(['inputTokens', 'outputTokens']));
  await assert.rejects(gateway.request(async () => ({ text: 'never' }), request),
    error => error instanceof BudgetExhausted && error.dimension === 'inputTokens');
});

test('RunStore rejects a live lock, removes a stale lock, and commits complete atomic writes', () => {
  const directory = temp();
  let store;
  try {
    store = new RunStore(directory);
    assert.throws(() => new RunStore(directory), /locked by process/);
    store.write('report.json', { revision: 1, payload: 'x'.repeat(200_000) });
    store.write('report.json', { revision: 2, payload: 'y'.repeat(200_000) });
    assert.equal(store.read('report.json').revision, 2);
    assert.ok(!readdirSync(directory).some(name => name.endsWith('.tmp')));
    assert.throws(() => store.write('report.json', { invalid: Number.NaN }), /finite JSON/);
    assert.equal(store.read('report.json').revision, 2, 'failed serialization leaves the prior file intact');
    store.close(); store = undefined;

    writeFileSync(join(directory, 'run.lock'), JSON.stringify({ pid: 2_147_483_647, token: 'stale' }));
    store = new RunStore(directory);
    assert.equal(store.read('run.lock').pid, process.pid);
  } finally { store?.close(); rmSync(directory, { recursive: true, force: true }); }
});

test('checkpoint resume requires the supported engine and exact search fingerprints', () => {
  const directory = temp(); let store;
  try {
    store = new RunStore(directory);
    const state = { schema: 'natlang.adaptation-run/v1', engine: ENGINE_VERSION, suiteHash: 'suite-a',
      programHash: 'program-a', optionsHash: 'options-a', runId: 'run', seed: 7, iteration: 2 };
    saveCheckpoint(store, state);
    assert.deepEqual(JSON.parse(JSON.stringify(readCheckpoint(store, 'suite-a', 'program-a', 'options-a'))), state);
    assert.throws(() => readCheckpoint(store, 'suite-b', 'program-a', 'options-a'), /fingerprint mismatch/);
    const saved = store.read('checkpoint.json');
    store.write('checkpoint.json', { ...saved, iteration: 999 });
    assert.throws(() => readCheckpoint(store, 'suite-a', 'program-a', 'options-a'), /integrity mismatch/);
    store.write('checkpoint.json', { ...state, engine: 'natlang.ax-gepa/older' });
    assert.throws(() => readCheckpoint(store, 'suite-a', 'program-a', 'options-a'), /unsupported optimization checkpoint/);
    store.write('checkpoint.json', { ...state, schema: 'natlang.adaptation-run/v0' });
    assert.throws(() => readCheckpoint(store, 'suite-a', 'program-a', 'options-a'), /unsupported optimization checkpoint/);
  } finally { store?.close(); rmSync(directory, { recursive: true, force: true }); }
});

test('priced bounded requests reserve concurrent capacity and record actual spend by role', async () => {
  assert.throws(() => new UsageGateway(limits({ maxCost: 1 })), /pricing and bounded requests/);
  const budget = limits({ maxCost: 0.000015, maxInputTokens: 20, maxOutputTokens: 10,
    pricing: { inputPerMillion: 1, outputPerMillion: 2 }, requestBounds: { maxInputTokens: 10, maxOutputTokens: 2 } });
  const gateway = new UsageGateway(budget);
  let release;
  const first = gateway.request(forwarded => {
    assert.equal(forwarded.max_tokens, 2);
    return new Promise(resolve => { release = resolve; });
  }, request, undefined, 'reflection');
  await Promise.resolve();
  assert.equal(gateway.snapshot().reserved.cost, 0.000014);
  await assert.rejects(gateway.request(() => ({ text: 'never' }), request), /cost \(reserved\)/);
  const crashed = new UsageGateway(budget, gateway.snapshot());
  assert.equal(crashed.ledger.unknownRequests, 1);
  assert.equal(crashed.ledger.usage.cost, null);
  assert.throws(() => crashed.check(), /inputTokens|cost/);
  release({ text: 'ok', prompt_tokens: 3, completion_tokens: 1 }); await first;
  assert.equal(gateway.ledger.usage.cost, 0.000005);
  assert.equal(gateway.ledger.roles.reflection.modelCalls, 1);
  assert.equal(gateway.snapshot().reserved.modelCalls, 0);
});
