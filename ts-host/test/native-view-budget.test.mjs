import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { ModelBudget } from '../dist/native/model-budget.js';
import { serverViewer } from '../dist/neuralese/view.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

test('batch reservation prevents siblings from borrowing its request or token allowance', async () => {
  const budget = new ModelBudget(3, 5), batch = await budget.reserveBatch();
  assert.deepEqual([batch.maxRequests, batch.maxTokens], [3, 5]);
  let resolved = false;
  const pending = budget.reserve(null).then(value => { resolved = true; return value; });
  await Promise.resolve(); assert.equal(resolved, false);
  batch.finish({ requests: 2, tokens: 3 });
  const next = await pending;
  assert.equal(next.limit, 2); next.finish(2);
  assert.equal(await budget.reserve(null), undefined);
});

test('failed multi-write reservations conservatively consume their bounded allowance', async () => {
  const budget = new ModelBudget(2, 4), batch = await budget.reserveBatch();
  batch.finish();
  assert.equal(await budget.reserve(null), undefined);
});

test('client sends allowance and reconciles verified partial view usage', async () => {
  let requestBody, received;
  const turn = { messages: [], tools: [], template: {}, response: { usage: { completion_tokens: 2 },
    x_natlang_rl_capture: { schema: 'natlang.native-rl-sampling/2', policy: {}, prompt: [], generated_inputs: [], writes: [] },
    native_state: { blocks: [] } } };
  const server = createServer(async (req, res) => {
    const chunks = []; for await (const chunk of req) chunks.push(chunk);
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/neuralese/info') return res.end(JSON.stringify({ dialects: ['native-state/test'],
      capabilities: ['view', 'view.budget/1'], native_state: { enabled: true } }));
    if (req.url === '/v1/neuralese/view') {
      requestBody = JSON.parse(Buffer.concat(chunks));
      return res.end(JSON.stringify({ budget_exhausted: true, budget_usage: { requests: 1, tokens: 2 }, source_turns: [turn] }));
    }
    res.statusCode = 404; res.end('{}');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const viewer = serverViewer({ endpoint: `http://127.0.0.1:${server.address().port}`, store: new MemoryNeuraleseStore() });
    assert.equal(await viewer({ value: 'some data' }, { maxRequests: 3, maxTokens: 5, onUsage: usage => { received = usage; } }), undefined);
    assert.deepEqual(requestBody.budget, { max_requests: 3, max_tokens: 5 });
    assert.deepEqual(received, { requests: 1, tokens: 2 });
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('a view shares the invocation budget and returns unused tokens to the model', async () => {
  const calls = [], store = new MemoryNeuraleseStore();
  const view = Object.assign(async (_site, budget) => {
    assert.deepEqual([budget.maxRequests, budget.maxTokens], [2, 3]);
    budget.onUsage({ requests: 1, tokens: 2 }); return undefined;
  }, { budgeted: true });
  const driver = Object.assign(async request => {
    calls.push(request);
    return { calls: [['return_result', { status: 'success', value: 7 }]], completion_tokens: 1 };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ calls: false, model: driver, limits: { maxModelRequests: 2, maxModelTokens: 3 },
    neuralese: { store, dialect: 'native-state/test', view } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {value: string}\nreturns: number\n---\nReturn 7.\n' }, 'run.nl');
  assert.equal(await runtime.run(() => fn('a'.repeat(100000))), 7);
  assert.equal(calls[0].max_tokens, 1);
});

test('unreported unbounded usage remains unknown, never a fabricated zero', async () => {
  const budget = new ModelBudget(undefined, 4), batch = await budget.reserveBatch();
  batch.finish();
  assert.deepEqual(budget.usage(), { requests: null, tokens: null });
  assert.equal(budget.hasRemaining(), false);
});

test('older view servers are refused before a bounded request and spend no allowance', async () => {
  let posts = 0, reported;
  const server = createServer(async (req, res) => {
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/neuralese/info') return res.end(JSON.stringify({ dialects: ['native-state/test'],
      capabilities: ['view'], native_state: { enabled: true } }));
    posts++; res.end('{}');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const viewer = serverViewer({ endpoint: `http://127.0.0.1:${server.address().port}`, store: new MemoryNeuraleseStore() });
    await assert.rejects(viewer({ value: 'x' }, { maxRequests: 2, maxTokens: 3, onUsage: usage => { reported = usage; } }), /budget-unavailable/);
    assert.equal(posts, 0); assert.deepEqual(reported, { requests: 0, tokens: 0 });
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('client refuses a budget receipt inconsistent with actual source turns', async () => {
  let reported = false;
  const server = createServer(async (req, res) => {
    for await (const _ of req) { /* drain */ }
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/neuralese/info') return res.end(JSON.stringify({ dialects: ['native-state/test'],
      capabilities: ['view', 'view.budget/1'], native_state: { enabled: true } }));
    res.end(JSON.stringify({ budget_exhausted: true, budget_usage: { requests: 1, tokens: 0 },
      source_turns: [{ response: { usage: { completion_tokens: 2 } } }] }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const viewer = serverViewer({ endpoint: `http://127.0.0.1:${server.address().port}`, store: new MemoryNeuraleseStore() });
    await assert.rejects(viewer({ value: 'x' }, { maxRequests: 2, maxTokens: 3, onUsage: () => { reported = true; } }), /budget-integrity/);
    assert.equal(reported, false);
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('forced text writes wait for a view reservation to release request capacity', async () => {
  const budget = new ModelBudget(2, 4), batch = await budget.reserveBatch();
  let resolved = false;
  const pending = budget.reserveForcedText().then(value => { resolved = true; return value; });
  await Promise.resolve(); assert.equal(resolved, false);
  batch.finish({ requests: 1, tokens: 1 });
  assert.equal(await pending, true);
  assert.equal(await budget.reserveForcedText(), false);
});

test('concurrent views receive disjoint remaining allowances', async () => {
  const budget = new ModelBudget(4, 6), first = await budget.reserveBatch();
  let resolved = false;
  const pending = budget.reserveBatch().then(value => { resolved = true; return value; });
  await Promise.resolve(); assert.equal(resolved, false);
  first.finish({ requests: 2, tokens: 3 });
  const second = await pending;
  assert.deepEqual([second.maxRequests, second.maxTokens], [2, 3]);
  second.finish({ requests: 1, tokens: 2 });
  assert.deepEqual(budget.usage(), { requests: 3, tokens: 5 });
});

test('malformed view sampling retains the raw turn and charges verified usage', async () => {
  let reported, retained;
  const raw = [{ response: { usage: { completion_tokens: 2 }, x_natlang_rl_capture: { schema: 'bad' } } }];
  const server = createServer(async (req, res) => {
    for await (const _ of req) { /* drain */ }
    res.setHeader('content-type', 'application/json');
    if (req.url === '/v1/neuralese/info') return res.end(JSON.stringify({ dialects: ['native-state/test'],
      capabilities: ['view', 'view.budget/1'], native_state: { enabled: true } }));
    res.end(JSON.stringify({ budget_error: 'bad receipt', budget_usage: { requests: 1, tokens: 2 }, source_turns: raw }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const viewer = serverViewer({ endpoint: `http://127.0.0.1:${server.address().port}`, store: new MemoryNeuraleseStore(),
      onSourceTurns: turns => { retained = turns; } });
    await assert.rejects(viewer({ value: 'x' }, { maxRequests: 2, maxTokens: 3, onUsage: usage => { reported = usage; } }),
      error => /source-integrity/.test(error.message) && error.native_source_turns.length === 1);
    assert.deepEqual(retained, raw); assert.deepEqual(reported, { requests: 1, tokens: 2 });
  } finally { await new Promise(resolve => server.close(resolve)); }
});
