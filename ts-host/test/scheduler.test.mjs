import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { chatCompletionModelTurn, requestLimit, limitedTransport } from '../dist/model/chat-completion.js';
import { createScheduler, prefixKey } from '../dist/model/scheduler.js';
import { coalescingScorer, explicitBatchScoreMany, concurrentScoreMany } from '../dist/model/scoring.js';
import { planServerSlots } from '../dist/model/server-slots.js';
import { resolveModelChoice } from '../dist/model/config.js';
import { scoreMany } from '../dist/native/decision.js';

const tick = (ms = 0) => new Promise(resolve => setTimeout(resolve, ms));
const reply = text => ({ choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: text } }] });
const turnRequest = (invocation, text = 'hi') => ({ ...(invocation ? { invocation_id: invocation } : {}),
  messages: [{ role: 'system', content: 'sys' }, { role: 'user', content: text }], tools: [], seed: null, max_tokens: null });

/** A transport whose requests finish when the test says so. */
function gatedTransport() {
  const log = [], gates = [];
  const transport = (body, _signal, meta) => new Promise(resolve => {
    const entry = { body, meta, started: log.length };
    log.push(entry);
    gates.push(() => resolve(reply(String(body.messages.at(-1).content))));
  });
  return { transport, log, finish: (index = 0) => gates[index]() };
}

test('requests that become ready together are released as one batch, a later request starts a new one', async () => {
  const scheduler = createScheduler({ maxConcurrent: 8 });
  const seen = [];
  const send = meta => scheduler.acquire({ ...meta, onScheduled: info => seen.push(info) }).then(release => release());
  await Promise.all([send(), send(), send(), send(), send()]);
  await send();
  assert.equal(seen.length, 6);
  assert.deepEqual(new Set(seen.slice(0, 5).map(item => item.batch_id)).size, 1);
  assert.ok(seen.slice(0, 5).every(item => item.batch_size === 5));
  assert.notEqual(seen[5].batch_id, seen[0].batch_id);
  assert.equal(seen[5].batch_size, 1);
  assert.equal(scheduler.occupancy().requests, 6);
  assert.equal(scheduler.occupancy().batches, 2);
});

test('an explicit-batch backend waits a coalescing window for company', async () => {
  const scheduler = createScheduler({ mode: 'explicit-batch', maxConcurrent: 8, coalesceMs: 15 });
  const seen = [];
  const send = () => scheduler.acquire({ onScheduled: info => seen.push(info) }).then(release => release());
  const first = send();
  await tick(5);
  const second = send();
  await Promise.all([first, second]);
  assert.deepEqual(seen.map(item => item.batch_size), [2, 2]);
  assert.equal(scheduler.capabilities.batching.mode, 'explicit-batch');
  assert.equal(scheduler.capabilities.batching.maxConcurrent, 8);
});

test('a turn of a running call is served before the first turn of a new call', async () => {
  const scheduler = createScheduler({ maxConcurrent: 1, coalesceMs: 0 });
  const order = [];
  const hold = await scheduler.acquire({ invocation_id: 'running' });   // turn 1 of "running" holds the only slot
  const waiting = [
    scheduler.acquire({ invocation_id: 'new-a' }).then(release => { order.push('new-a'); release(); }),
    scheduler.acquire({ invocation_id: 'new-b' }).then(release => { order.push('new-b'); release(); }),
    scheduler.acquire({ invocation_id: 'running' }).then(release => { order.push('running turn 2'); release(); }),
  ];
  await tick(5);
  assert.equal(scheduler.queued, 3);
  hold();
  await Promise.all(waiting);
  assert.deepEqual(order, ['running turn 2', 'new-a', 'new-b']);
  assert.deepEqual(scheduler.records().map(item => item.priority), ['new', 'running', 'new', 'new']);
});

test('requests with the same prompt prefix are sent next to each other', async () => {
  const scheduler = createScheduler({ maxConcurrent: 1 });
  const order = [];
  const hold = await scheduler.acquire({ group: 'x' });
  const ask = (group, label) => scheduler.acquire({ group }).then(release => { order.push(label); release(); });
  const all = [ask('x', 'x1'), ask('y', 'y1'), ask('x', 'x2'), ask('y', 'y2')];
  await tick(5); hold();
  await Promise.all(all);
  assert.deepEqual(order, ['x1', 'x2', 'y1', 'y2']);
  assert.equal(prefixKey({ messages: [{ role: 'system', content: 'a' }] }), prefixKey({ messages: [{ role: 'system', content: 'a' }, { role: 'user', content: 'b' }] }));
  assert.notEqual(prefixKey({ messages: [{ role: 'system', content: 'a' }] }), prefixKey({ messages: [{ role: 'system', content: 'b' }] }));
});

test('a queued request is cancelled by its signal or by its invocation, and is never sent', async () => {
  const scheduler = createScheduler({ maxConcurrent: 1 });
  const sent = [];
  const hold = await scheduler.acquire({ invocation_id: 'hold' });
  const controller = new AbortController();
  const aborted = scheduler.acquire({ invocation_id: 'a' }, controller.signal).then(() => sent.push('a'));
  const cancelled = scheduler.acquire({ invocation_id: 'b' }).then(() => sent.push('b'));
  const kept = scheduler.acquire({ invocation_id: 'c' }).then(release => { sent.push('c'); release(); });
  await tick(5);
  controller.abort(new Error('stop a'));
  assert.equal(scheduler.cancel('b', new Error('stop b')), 1);
  await assert.rejects(aborted, /stop a/);
  await assert.rejects(cancelled, /stop b/);
  assert.equal(scheduler.queued, 1);
  hold();
  await kept;
  assert.deepEqual(sent, ['c']);
  assert.equal(scheduler.inFlight, 0);
  await assert.rejects(scheduler.acquire({}, AbortSignal.abort(new Error('already'))), /already/);
  scheduler.close();
  await assert.rejects(scheduler.acquire({}), /closed/);
});

test('a scheduler without priority, adjacency or window admits requests exactly as requestLimit does', async () => {
  const arrivals = Array.from({ length: 40 }, (_, index) => ({ id: index, delay: (index * 7) % 5, hold: 1 + (index * 3) % 4,
    invocation: `call-${index % 6}`, group: `g${index % 3}` }));
  const run = async acquire => {
    let active = 0, peak = 0;
    const started = [], finished = [];
    await Promise.all(arrivals.map(async item => {
      await tick(item.delay);
      const release = await acquire(item);
      active++; peak = Math.max(peak, active); started.push(item.id);
      await tick(item.hold);
      active--; finished.push(item.id); release();
    }));
    return { started, peak, finished: finished.length };
  };
  for (const size of [1, 3]) {
    const limit = requestLimit(size);
    const scheduler = createScheduler({ maxConcurrent: size, priority: false, adjacency: false, coalesceMs: 0 });
    const plain = await run(() => limit.acquire());
    const scheduled = await run(item => scheduler.acquire({ invocation_id: item.invocation, group: item.group }));
    assert.equal(plain.peak, size);
    assert.equal(scheduled.peak, size);
    assert.equal(scheduled.finished, arrivals.length);
    // Both are first-come-first-served: every request that waited is admitted in the order it arrived.
    const byArrival = [...arrivals].sort((a, b) => a.delay - b.delay || a.id - b.id).map(item => item.id);
    assert.equal(scheduled.started.length, plain.started.length);
    const rank = new Map(byArrival.map((id, index) => [id, index]));
    const inversions = ids => ids.reduce((count, id, index) => count + ids.slice(index + 1).filter(later => rank.get(later) < rank.get(id)).length, 0);
    assert.ok(inversions(scheduled.started) <= inversions(plain.started) + 3, `size ${size}: ${inversions(scheduled.started)} vs ${inversions(plain.started)}`);
  }
  const asLimit = createScheduler({ maxConcurrent: 2 }).asRequestLimit();
  assert.equal(asLimit.size, 2);
  const release = await asLimit.acquire();
  release();
});

test('turns through a scheduler carry their batch fields, and the trace order matches an unscheduled run', async () => {
  const program = async transport => {
    const events = [];
    const driver = chatCompletionModelTurn(transport, { onRequestStart: (request, retry) => { events.push(`start ${request.messages.at(-1).content}#${retry}`); } });
    for (const text of ['one', 'two', 'three']) {
      const turn = await driver(turnRequest('call-1', text));
      events.push(`end ${turn.text}`);
    }
    const concurrent = await Promise.all(['a', 'b', 'c'].map(text => driver(turnRequest(`call-${text}`, text))));
    events.push(...concurrent.map(turn => `joined ${turn.text}`));
    return { events, last: concurrent };
  };
  const plainTransport = async body => reply(String(body.messages.at(-1).content));
  const baseline = await program(plainTransport);
  const scheduler = createScheduler({ maxConcurrent: 2 });
  const scheduled = await program(scheduler.transport(plainTransport));
  assert.deepEqual(scheduled.events, baseline.events);
  assert.equal(baseline.last[0].scheduling, undefined);
  const info = scheduled.last[0].scheduling;
  assert.ok(info.batch_id && info.batch_size >= 1 && info.in_flight >= 1 && info.queue_wait_ms >= 0);
  assert.equal(scheduler.records().length, 6);
  // turns 2 and 3 of call-1 are running-call turns
  assert.deepEqual(scheduler.records().slice(0, 3).map(item => item.priority), ['new', 'running', 'running']);
});

test('a streamed reply holds its slot until it has been read', async () => {
  const scheduler = createScheduler({ maxConcurrent: 1 });
  const transport = scheduler.transport(async () => (async function* () { yield { choices: [{ delta: { content: 'x' } }] }; })());
  const stream = await transport({ messages: [] });
  assert.equal(scheduler.inFlight, 1);
  for await (const _ of stream) { /* drain */ }
  assert.equal(scheduler.inFlight, 0);
});

test('requests go out in priority order through a gated transport', async () => {
  const scheduler = createScheduler({ maxConcurrent: 1 });
  const gated = gatedTransport();
  const driver = chatCompletionModelTurn(scheduler.transport(gated.transport));
  const first = driver(turnRequest('r', 'r1'));
  await tick(5);
  gated.finish(0); await first;
  const holder = driver(turnRequest('h', 'h1'));      // takes the slot while the others queue
  await tick(5);
  const fresh = driver(turnRequest('n', 'n1'));
  const running = driver(turnRequest('r', 'r2'));     // turn 2 of an invocation the scheduler has seen
  await tick(5);
  gated.finish(1); await holder; await tick(5);
  assert.deepEqual(gated.log.map(item => item.body.messages.at(-1).content), ['r1', 'h1', 'r2']);
  gated.finish(2); await running; await tick(5);
  gated.finish(3); await fresh;
  assert.deepEqual(gated.log.map(item => item.body.messages.at(-1).content), ['r1', 'h1', 'r2', 'n1']);
});

test('limitedTransport passes request metadata through unchanged', async () => {
  let seenMeta;
  const transport = limitedTransport(async (_body, _signal, meta) => { seenMeta = meta; return reply('ok'); }, requestLimit(1));
  await chatCompletionModelTurn(transport)(turnRequest('call-9'));
  assert.equal(seenMeta.invocation_id, 'call-9');
});

test('decision readouts that arrive together are scored by one scoreMany; failures stay with their caller', async () => {
  const batches = [];
  const base = Object.assign(async () => { throw new Error('single path must not be used'); }, {
    scoreMany: async items => { batches.push(items.length);
      return items.map(item => item.options[0] === 'bad' ? { status: 'rejected', reason: new Error('bad item') } : { status: 'fulfilled', value: { log_probs: item.options.map((_, i) => -(i + 1)) } }); } });
  const scorer = coalescingScorer(base);
  const results = await Promise.allSettled([scorer({ messages: [], options: ['a', 'b'] }), scorer({ messages: [], options: ['bad'] }), scorer({ messages: [], options: ['c'] })]);
  assert.deepEqual(batches, [3]);
  assert.deepEqual(results.map(item => item.status), ['fulfilled', 'rejected', 'fulfilled']);
  assert.deepEqual(results[0].value.log_probs, [-1, -2]);
  assert.equal(results[0].value.batch.batch_size, 3, 'a coalesced readout records its batch');
  assert.equal(results[0].value.batch.batch_id, results[2].value.batch.batch_id);
  const controller = new AbortController();
  const aborted = scorer({ messages: [], options: ['a'] }, controller.signal);
  const other = scorer({ messages: [], options: ['a'] });
  controller.abort(new Error('gone'));
  await assert.rejects(aborted, /gone/);
  assert.equal((await other).log_probs.length, 1);
  assert.deepEqual(batches, [3, 1], "an aborted caller is dropped from its batch");
});

test('scoreMany falls back to concurrent single scoring', async () => {
  let active = 0, peak = 0;
  const scorer = async ({ options }) => { active++; peak = Math.max(peak, active); await tick(2); active--; return { log_probs: options.map(() => -1) }; };
  const results = await scoreMany(scorer, [1, 2, 3, 4].map(() => ({ messages: [], options: ['x', 'y'] })));
  assert.equal(results.length, 4);
  assert.equal(peak, 4);
  assert.ok(results.every(item => item.status === 'fulfilled'));
  const through = createScheduler({ maxConcurrent: 2 });
  let slots = 0, held = 0;
  const limited = async () => { const release = await through.acquire(); slots++; held = Math.max(held, through.inFlight); await tick(2); release(); return { log_probs: [0] }; };
  await concurrentScoreMany(limited)([{ messages: [], options: ['x'] }, { messages: [], options: ['x'] }, { messages: [], options: ['x'] }]);
  assert.equal(slots, 3);
  assert.equal(held, 2);
});

test('explicit-batch scoring posts prefix and continuations once, then falls back when the endpoint is missing', async () => {
  const posts = [];
  let mode = 'ok';
  const instance = createServer((incoming, response) => {
    const chunks = [];
    incoming.on('data', chunk => chunks.push(chunk));
    incoming.on('end', () => {
      posts.push(JSON.parse(Buffer.concat(chunks).toString()));
      if (mode === 'missing') { response.writeHead(404); response.end('no'); return; }
      const body = posts.at(-1);
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ results: body.items.map(item => item.continuations[0] === 'boom' ? { error: 'too long' } :
        { log_probs: item.continuations.map((_, index) => -index), tokens: item.continuations.map(() => 2) }) }));
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${instance.address().port}/v1/natlang/score`;
  const fallbackCalls = [];
  const fallback = async items => { fallbackCalls.push(items.length); return items.map(item => ({ status: 'fulfilled', value: { log_probs: item.options.map(() => 0) } })); };
  try {
    const many = explicitBatchScoreMany({ url, extra: { model: 'm' }, scheduler: createScheduler({ mode: 'explicit-batch', maxConcurrent: 1 }) }, fallback);
    const items = [{ messages: [{ role: 'user', content: 'p' }], options: ['a', 'b', 'c'] }, { messages: [], options: ['boom', 'z'] }];
    const results = await many(items);
    assert.equal(posts.length, 1);
    assert.equal(posts[0].model, 'm');
    assert.deepEqual(posts[0].items[0], { messages: [{ role: 'user', content: 'p' }], continuations: ['a', 'b', 'c'] });
    assert.deepEqual(results[0].value, { log_probs: [0, -1, -2], tokens: [2, 2, 2] });
    assert.equal(results[1].status, 'rejected');
    assert.deepEqual(fallbackCalls, []);
    mode = 'missing';
    const after = await many(items);
    assert.deepEqual(fallbackCalls, [2]);
    assert.ok(after.every(item => item.status === 'fulfilled'));
    await many(items);
    assert.equal(posts.length, 2, 'a missing endpoint is not asked again');
    assert.deepEqual(fallbackCalls, [2, 2]);
  } finally { await new Promise(resolve => instance.close(resolve)); }
});

test('server slots follow the memory formula unless configured; the default KV budget is capped at 4 GiB', () => {
  const GiB = 2 ** 30;
  const base = { modelBytes: 1 * GiB, defaultContextTokens: 8192, environment: {} };   // 0.5 GiB of KV per slot
  // default: min(free / 2, 4 GiB) of KV alone
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 1024 * GiB }).slots, 8);
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 1024 * GiB }).budgetBytes, 4 * GiB);
  const small = planServerSlots({ ...base, availableMemoryBytes: 4 * GiB });
  assert.equal(small.slots, 4);
  assert.equal(small.totalContext, 32768);
  assert.equal(small.budgetSource, 'default');
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 0.5 * GiB }).slots, 1);
  // 16K context at 64 KiB/token is 1 GiB per slot: about 4 slots under the cap, however much memory is free
  const wide = planServerSlots({ ...base, defaultContextTokens: 16384, availableMemoryBytes: 512 * GiB });
  assert.equal(wide.slots, 4);
  assert.match(wide.formula, /4294967296/);
  // explicit settings win and are not capped
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 64 * GiB, local: { parallel: 3, contextTokens: 4096 } }).slots, 3);
  const explicit = planServerSlots({ ...base, availableMemoryBytes: 1 * GiB, local: { memoryBudgetMiB: 2048 } });
  assert.equal(explicit.slots, 2);
  assert.equal(explicit.budgetSource, 'memoryBudgetMiB');
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 1 * GiB, local: { memoryBudgetMiB: 20 * 1024 } }).slots, 8);
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 64 * GiB, local: { kvBytesPerToken: 2 ** 20, memoryBudgetMiB: 4096 } }).slots, 1);
  assert.equal(planServerSlots({ ...base, availableMemoryBytes: 64 * GiB, environment: { NATLANG_MODEL_MEMORY_BUDGET_MIB: '3072' } }).slots, 4);
});

test('batching settings are validated and kept off the executor identity', async () => {
  const choice = resolveModelChoice({ batching: { mode: 'explicit-batch', maxConcurrent: 4, coalesceMs: 3, scoreEndpoint: true }, local: { memoryBudgetMiB: 4096 } });
  assert.equal(choice.batching.maxConcurrent, 4);
  assert.throws(() => resolveModelChoice({ batching: { mode: 'fast' } }), /batching.mode/);
  assert.throws(() => resolveModelChoice({ batching: { maxConcurrent: 0 } }), /maxConcurrent/);
  assert.throws(() => resolveModelChoice({ batching: { scoreEndpoint: true } }), /explicit-batch/);
  assert.throws(() => resolveModelChoice({ local: { kvBytesPerToken: 0 } }), /kvBytesPerToken/);
  assert.throws(() => resolveModelChoice({ local: { args: ['-np', '2'] } }), /managed flag/);
  const { executorIdentityForChoice } = await import('../dist/model/config.js');
  assert.equal(JSON.stringify(executorIdentityForChoice(choice)).includes('batching'), false);
});
