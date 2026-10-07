import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { KvBudget, collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, expectedProvenance, jobKey,
  loadRecords, nativeJobRunner, recordDigest, withExecutionPlans } from '../dist/teacher/collector.js';

const record = id => ({ version: 'natlang.program/2', id, kind: 'lambda_source', source: 'fixture',
  split: 'test', source_ids: [id], source_groups: [id], license: 'test', semantics: {
    root: 'one.nl', files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' }, inputs: {}, expected: 1,
    operation: 'exact' } });
const config = (dir, surface = 'surface-a') => ({ jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'),
  workers: 2, modelId: 'teacher', rootSeed: 7, systemPrompt: 'prompt', contextTokens: 16384,
  toolSurfaceSha256: surface });
const row = (item, provenance) => ({ version: 'test', task: { program_ir: item.record }, provenance,
  outcome: { status: 'done', accepted: true } });

test('execution planning is opt-in, conditions the action, and replaces provider reasoning', async () => {
  const requests = [];
  const send = async request => {
    requests.push(structuredClone(request));
    if (requests.length === 1) return { calls: [['execution_plan', { plan: 'Read n, then compute exactly.' }]],
      completion_tokens: 7 };
    return { calls: [['eval', { code: 'n + 1' }]], reasoning: 'opaque provider trace', completion_tokens: 5 };
  };
  const request = { messages: [{ role: 'user', content: 'Increment n.' }], tools: [{ type: 'function', function: {
    name: 'eval', parameters: { type: 'object' } } }], seed: 9, max_tokens: 100 };
  const result = await withExecutionPlans(send)(request);
  assert.equal(requests[0].tool_choice, 'required');
  assert.deepEqual(requests[0].tools.map(tool => tool.function.name), ['execution_plan']);
  assert.deepEqual(requests[1].messages.slice(-2).map(message => message.role), ['assistant', 'tool']);
  assert.equal(JSON.parse(requests[1].messages.at(-2).tool_calls[0].function.arguments).plan,
    'Read n, then compute exactly.');
  assert.deepEqual(requests[1].tools, request.tools);
  assert.equal(result.execution_plan, 'Read n, then compute exactly.');
  assert.equal(result.reasoning, result.execution_plan);
  assert.equal(result.completion_tokens, 12);
});

test('execution planning falls through when required tools are unsupported', async () => {
  const requests = [];
  const send = async request => {
    requests.push(structuredClone(request));
    if (request.tools[0].function.name === 'execution_plan') throw new Error('tool_choice is unsupported');
    return { calls: [['eval', { code: 'n + 1' }]], reasoning: 'provider trace', completion_tokens: 5 };
  };
  const request = { messages: [{ role: 'user', content: 'Increment n.' }], tools: [{ type: 'function', function: {
    name: 'eval', parameters: { type: 'object' } } }], seed: 9, max_tokens: 100 };
  const result = await withExecutionPlans(send)(request);
  assert.equal(requests.length, 2);
  assert.deepEqual(requests[1], request, 'the ordinary action proceeds unchanged');
  assert.equal(result.execution_plan, null);
  assert.equal(result.reasoning, undefined, 'opaque provider reasoning is not retained in planning mode');
  assert.equal(result.calls[0][0], 'eval');
});

test('a rate limit while planning fails the turn for a retry instead of dropping its plan', async () => {
  const send = async request => {
    if (request.tools[0].function.name === 'execution_plan') throw new Error('{"detail":"Rate limit exceeded"}');
    return { calls: [['eval', { code: 'n + 1' }]], completion_tokens: 5 };
  };
  const request = { messages: [{ role: 'user', content: 'Increment n.' }], tools: [{ type: 'function', function: {
    name: 'eval', parameters: { type: 'object' } } }], seed: 9, max_tokens: 100 };
  await assert.rejects(withExecutionPlans(send)(request), /Rate limit exceeded/);
});

test('focused loader keeps source indexes and selects an exact range', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-load-')), path = join(dir, 'ir.jsonl');
  await writeFile(path, Array.from({ length: 12 }, (_, index) => JSON.stringify(record(`r${index}`))).join('\n') + '\n');
  const loaded = await loadRecords(path, 1, 10);
  assert.deepEqual(loaded.map(item => item.index), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.match(jobKey(loaded[0]), /^000001-[0-9a-f]{16}$/);
  assert.equal(recordDigest(loaded[0].record).length, 64);
});

test('all-mode permits an empty hard-state queue without inventing a teacher job', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-empty-')), path = join(dir, 'empty.jsonl');
  await writeFile(path, '');
  assert.deepEqual(await loadRecords(path, 0, 0), []);
  const options = config(dir);
  const result = await collectBatch([], options, async () => { throw new Error('must not run'); });
  assert.deepEqual(result, { completed: 0, missing: [] });
  assert.equal(await readFile(options.output, 'utf8'), '');
});

test('staggered workers start their first jobs apart', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-stagger-'));
  const records = [record('a'), record('b'), record('c')].map((value, index) => ({ index, record: value }));
  const started = [], t0 = Date.now();
  await collectBatch(records, { ...config(dir), workers: 3, workerStaggerMs: 40 }, async (item, provenance) => {
    started.push(Date.now() - t0); await new Promise(resolve => setTimeout(resolve, 150)); return row(item, provenance);
  });
  // Setup includes asynchronous filesystem work; measure staggering from the
  // first actual job instead of imposing a wall-clock ceiling on startup.
  assert.ok(started[1] - started[0] >= 35 && started[2] - started[0] >= 75, `start times ${started}`);
});

test('parallel completion merges in source order and matching jobs resume without calls', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-resume-'));
  const records = [record('first'), record('second'), record('third')].map((value, index) => ({ index, record: value }));
  let calls = 0;
  await collectBatch(records, config(dir), async (item, provenance) => {
    calls++; await new Promise(resolve => setTimeout(resolve, (2 - item.index) * 8)); return row(item, provenance);
  });
  const merged = (await readFile(config(dir).output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(merged.map(item => item.task.program_ir.id), ['first', 'second', 'third']);
  await collectBatch(records, config(dir), async () => { throw new Error('resume called the model'); });
  assert.equal(calls, 3);
});

test('provenance invalidates old jobs and partial results remain resumable', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-provenance-'));
  const records = [record('first'), record('second')].map((value, index) => ({ index, record: value }));
  await collectBatch(records, config(dir), async (item, provenance) => {
    if (item.index === 1) throw new Error('fixture failure'); return row(item, provenance);
  });
  assert.equal((await readFile(config(dir).output, 'utf8')).trim().split('\n').length, 1);
  let resumed = 0;
  await collectBatch(records, config(dir), async (item, provenance) => { resumed++; return row(item, provenance); });
  assert.equal(resumed, 1);
  let invalidated = 0;
  await collectBatch(records, config(dir, 'surface-b'), async (item, provenance) => {
    invalidated++; return row(item, provenance);
  });
  assert.equal(invalidated, 2);
  const provenance = expectedProvenance(records[0].record, config(dir, 'surface-b'));
  assert.equal(provenance.tool_surface_sha256, 'surface-b');
  assert.equal(provenance.runtime, 'typescript-native');
});

test('finished rows of an earlier run stand in for the same programs at other indexes', async () => {
  const earlier = await mkdtemp(join(tmpdir(), 'teacher-reuse-a-')), later = await mkdtemp(join(tmpdir(), 'teacher-reuse-b-'));
  const old = [record('kept'), record('stale'), record('migrated')].map((value, index) => ({ index, record: value }));
  await collectBatch(old, config(earlier), async (item, provenance) => row(item, provenance));
  const rows = (await readFile(config(earlier).output, 'utf8')).trim().split('\n').map(JSON.parse);
  rows[1].provenance.model = 'other-teacher';
  rows[2].provenance.tool_surface_sha256 = 'surface-old'; rows[2].provenance.finish_surface_migration = 'return_result-status/1';
  rows.push({ ...rows[0], task: { program_ir: record('rolled') }, provenance: { ...rows[0].provenance, program_ir_sha256: undefined },
    trajectory: [{ phase: 'checkpoint', model_response: {} }] });
  const source = join(earlier, 'earlier.jsonl');
  await writeFile(source, rows.map(value => JSON.stringify(value)).join('\n') + '\n');
  // The same programs sit at new indexes, next to a new one, in a later shard.
  rows[3].provenance.program_ir_sha256 = expectedProvenance({ index: 0, record: record('rolled') }.record, config(earlier)).program_ir_sha256;
  const shard = [record('new'), record('migrated'), record('stale'), record('kept'), record('rolled')].map((value, index) => ({ index, record: value }));
  const ran = [];
  const options = { ...config(later), reuse: [source] };
  await collectBatch(shard, options, async (item, provenance) => { ran.push(item.record.id); return row(item, provenance); });
  assert.deepEqual(ran.sort(), ['new', 'rolled', 'stale'], 'rows from another model or with a rollover checkpoint are collected again');
  const merged = (await readFile(options.output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(merged.map(value => value.task.program_ir.id), ['new', 'migrated', 'stale', 'kept', 'rolled']);
  assert.equal(merged[3].provenance.reused_from.path, source);
  assert.equal(merged[1].provenance.reused_from.provenance.tool_surface_sha256, 'surface-old');
  await collectBatch(shard, options, async () => { throw new Error('reused rows were collected again'); });
});

test('a row reused once is judged by where it was collected when reused again', async () => {
  const first = await mkdtemp(join(tmpdir(), 'teacher-rereuse-a-')), second = await mkdtemp(join(tmpdir(), 'teacher-rereuse-b-')),
    third = await mkdtemp(join(tmpdir(), 'teacher-rereuse-c-'));
  const shard = [record('only')].map((value, index) => ({ index, record: value }));
  await collectBatch(shard, config(first, 'surface-origin'), async (item, provenance) => row(item, provenance));
  // Reused into a run on surface-b, then again into a run on surface-c: only the origin surface is declared.
  await collectBatch(shard, { ...config(second, 'surface-b'), reuse: [config(first).output], reuseSurfaces: ['surface-origin'] },
    async () => { throw new Error('collected instead of reused'); });
  await collectBatch(shard, { ...config(third, 'surface-c'), reuse: [config(second).output], reuseSurfaces: ['surface-origin'] },
    async () => { throw new Error('collected instead of reused'); });
  const merged = JSON.parse((await readFile(config(third).output, 'utf8')).trim());
  assert.equal(merged.provenance.reused_from.path, config(first).output, 'reused_from names the run that collected it');
  assert.equal(merged.provenance.reused_from.provenance.tool_surface_sha256, 'surface-origin');
});

test('a later row that does not qualify never hides an earlier one, and declared surfaces are reused', async () => {
  const earlier = await mkdtemp(join(tmpdir(), 'teacher-mask-a-')), later = await mkdtemp(join(tmpdir(), 'teacher-mask-b-'));
  const old = [record('one'), record('two')].map((value, index) => ({ index, record: value }));
  await collectBatch(old, config(earlier, 'surface-old'), async (item, provenance) => row(item, provenance));
  const rows = (await readFile(config(earlier).output, 'utf8')).trim().split('\n').map(JSON.parse);
  const good = join(earlier, 'good.jsonl'), masking = join(earlier, 'masking.jsonl');
  await writeFile(good, rows.map(value => JSON.stringify(value)).join('\n') + '\n');
  await writeFile(masking, JSON.stringify({ ...rows[0], provenance: { ...rows[0].provenance, model: 'other' } }) + '\n');
  const ran = [];
  const runner = async (item, provenance) => { ran.push(item.record.id); return row(item, provenance); };
  await collectBatch(old, { ...config(later), reuse: [good, masking] }, runner);
  assert.deepEqual(ran.sort(), ['one', 'two'], 'rows of another tool surface are not reused unless declared');
  const again = await mkdtemp(join(tmpdir(), 'teacher-mask-c-'));
  ran.length = 0;
  await collectBatch(old, { ...config(again), reuse: [good, masking], reuseSurfaces: ['surface-old'] }, runner);
  assert.deepEqual(ran, [], 'declared surface rows are reused, and the later disqualified row does not mask one');
});

test('a row collected under another turn limit is reused only if it stayed clear of both limits', async () => {
  const earlier = await mkdtemp(join(tmpdir(), 'teacher-turns-a-')), later = await mkdtemp(join(tmpdir(), 'teacher-turns-b-'));
  const shard = [record('short'), record('long')].map((value, index) => ({ index, record: value }));
  await collectBatch(shard, { ...config(earlier), maxTurns: 30 }, async (item, provenance) =>
    ({ ...row(item, provenance), trajectory: Array.from({ length: item.index ? 17 : 16 }, () => ({ phase: 'action' })) }));
  const ran = [];
  await collectBatch(shard, { ...config(later), maxTurns: 20, reuse: [config(earlier).output] },
    async (item, provenance) => { ran.push(item.record.id); return row(item, provenance); });
  assert.deepEqual(ran, ['long'], 'a run 4 turns from the new limit would have been told the turns left');
});

test('native collector journals model replies and replays them after an interrupted request', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-turn-resume-'));
  let requests = 0, interrupted = false;
  const replies = [
    ['eval', { code: 'const answer: number = 1; return answer' }],
    null,
  ];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests++;
      if (!interrupted && requests === 2) { interrupted = true; request.socket.destroy(); return; }
      const logical = requests === 1 ? 0 : requests - 2;
      const reply = replies[logical];
      const message = reply ? { role: 'assistant', content: '', tool_calls: [{ id: `c${logical}`,
        type: 'function', function: { name: reply[0], arguments: JSON.stringify(reply[1]) } }] } :
        { role: 'assistant', content: '' };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const port = server.address().port, item = { index: 0, record: record('resumable-turns') };
    const options = { ...config(dir), workers: 1, endpoint: `http://127.0.0.1:${port}`,
      systemPrompt: defaultSystemPrompt, toolSurfaceSha256: await defaultToolSurfaceHash(),
      transportRetries: 1, retryDelayMs: 0 };
    const result = await collectBatch([item], options, nativeJobRunner(options));
    assert.equal(result.completed, 1);
    assert.equal(requests, 3, 'the first decoded response must be replayed locally after interruption');
    const output = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(output.outcome.accepted, true);
    assert.equal(output.trajectory.length, 2);
    await assert.rejects(readFile(join(options.jobs, `${jobKey(item)}.partial.json`)), /ENOENT/);
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('requests wait for room in the shared KV buffer, first come first served, and an oversized one runs alone', async () => {
  const kv = new KvBudget(100), order = [];
  await kv.acquire(60); order.push('a');
  const b = kv.acquire(50).then(() => order.push('b'));
  const c = kv.acquire(10).then(() => order.push('c'));
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.deepEqual(order, ['a'], 'b does not fit beside a, and c waits behind b');
  kv.release(60); await b; await c;
  assert.deepEqual(order, ['a', 'b', 'c']);
  kv.release(50); kv.release(10);
  await kv.acquire(500); order.push('huge');
  assert.deepEqual(order.at(-1), 'huge', 'a request larger than the buffer runs when nothing else is in flight');
  kv.release(500);
});


test('the native collector records the actions of child nl calls in the ledger, by their own call', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-child-ledger-'));
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      const { messages } = JSON.parse(body);
      // The root evaluates once, which stages its result, and then replies done.
      const child = String(messages[1].content).includes('nl@');
      const started = messages.some(message => message.role === 'tool' && String(message.content).includes('Staged'));
      const reply = child ? ['return_result', { status: 'success', value: true }] :
        started ? null : ['eval', { code: 'const odd = await nl<boolean>`Is value odd?`(1); return odd ? 1 : 0;' }];
      const message = reply ? { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function',
        function: { name: reply[0], arguments: JSON.stringify(reply[1]) } }] } : { role: 'assistant', content: '' };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const item = { index: 0, record: record('child-ledger') };
    const options = { ...config(dir), workers: 1, endpoint: `http://127.0.0.1:${server.address().port}`,
      systemPrompt: defaultSystemPrompt, toolSurfaceSha256: await defaultToolSurfaceHash(), transportRetries: 1, retryDelayMs: 0 };
    await collectBatch([item], options, nativeJobRunner(options));
    const output = JSON.parse((await readFile(options.output, 'utf8')).trim());
    const ledger = output.outcome.action_ledger;
    const calls = new Set(ledger.map(event => event.call_id));
    assert.equal(calls.size, 2, 'the root call and its child each have actions');
    assert.ok(ledger.some(event => event.name === 'return_result' && event.arguments.value === true), 'the child finish is logged');
    const bounded = { ...options, jobs: join(dir, 'bounded-jobs'), output: join(dir, 'bounded.jsonl'),
      maxTurns: 2, modelConcurrency: 1, maxModelRequests: 1 };
    const stopped = await collectBatch([item], bounded, nativeJobRunner(bounded));
    assert.deepEqual(stopped.missing, [0], 'the root and child share one model request budget');
    assert.match(await readFile(join(bounded.jobs, '000000.error.json'), 'utf8'), /whole-case model request budget/);
    assert.equal(await readFile(bounded.output, 'utf8'), '', 'budget exhaustion is never admitted as a result');
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('whole-case request exhaustion aborts nested execution while preserving the resumable partial', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-budget-abort-'));
  let requests = 0, runSignal;
  const server = createServer((request, response) => {
    requests++;
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'continue' } }],
      usage: { prompt_tokens: 10, completion_tokens: 2 } }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const item = { index: 0, record: record('budget-abort') };
    const options = { ...config(dir), workers: 1, endpoint: `http://127.0.0.1:${server.address().port}`,
      systemPrompt: defaultSystemPrompt, maxModelRequests: 1, transportRetries: 0,
      execution: { identity: 'budget-abort-test/1', run: async (_record, driver, runtime) => {
        runSignal = runtime.signal;
        const request = { messages: [{ role: 'user', content: 'one request' }], tools: [] };
        await driver(request);
        await assert.rejects(driver(request), error => error.code === 'NATLANG_MODEL_REQUEST_BUDGET');
        assert.equal(runtime.signal.aborted, true, 'the budget error aborts the whole execution tree');
        await assert.rejects(driver(request), error => error.code === 'NATLANG_MODEL_REQUEST_BUDGET',
          'subsequent child or parent retries see the terminal cap without starting another request');
        return { trace: [] };
      } } };
    const result = await collectBatch([item], options, nativeJobRunner(options));
    assert.deepEqual(result.missing, [0]);
    assert.equal(requests, 1, 'only the in-budget request reaches the model');
    assert.equal(runSignal.aborted, true);
    assert.match(await readFile(join(options.jobs, '000000.error.json'), 'utf8'), /whole-case model request budget/);
    const partial = JSON.parse(await readFile(join(options.jobs, `${jobKey(item)}.partial.json`), 'utf8'));
    assert.equal(partial.turns.length, 1, 'the completed response remains available for an explicit resume');
    assert.equal(await readFile(options.output, 'utf8'), '', 'budget exhaustion does not produce an admitted result');
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('a teacher cannot self-grade and the judge shares the whole-case request budget', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-judge-budget-'));
  const base = { ...config(dir), workers: 1, endpoint: 'http://127.0.0.1:1',
    judgeModel: { modelId: 'teacher', endpoint: 'http://127.0.0.1:1' } };
  assert.throws(() => nativeJobRunner(base), /distinct model IDs/);
  const server = createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '',
      tool_calls: [{ id: 'c1', type: 'function', function: { name: 'return_result',
        arguments: JSON.stringify({ status: 'success', value: 1 }) } }] } }],
      usage: { prompt_tokens: 10, completion_tokens: 4 } }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const item = { index: 0, record: record('judge-budget') };
    item.record.semantics.oracle = { level: 'judged', rubric: 'Return one.' };
    const options = { ...base, endpoint: `http://127.0.0.1:${server.address().port}`,
      judgeModel: { modelId: 'independent-judge', endpoint: 'http://127.0.0.1:1' },
      maxModelRequests: 1, maxTurns: 2, transportRetries: 1 };
    const result = await collectBatch([item], options, nativeJobRunner(options));
    assert.deepEqual(result.missing, [0]);
    assert.match(await readFile(join(options.jobs, '000000.error.json'), 'utf8'), /whole-case model request budget/);
    assert.equal(await readFile(options.output, 'utf8'), '');
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('parallel judgments complete plan/action pairs and retain durable progress at the request cap', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-paired-plans-'));
  const kinds = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      const payload = JSON.parse(body), planning = payload.tools[0].function.name === 'execution_plan';
      kinds.push(planning ? 'plan' : 'action');
      const child = String(payload.messages[1].content).includes('nl@');
      const started = payload.messages.some(m => m.role === 'tool' && String(m.content).includes('Staged'));
      const call = planning ? ['execution_plan', { plan: 'Read the argument and return one; aggregate in the root.' }] :
        child ? ['return_result', { status: 'success', value: 1 }] : started ? ['return_result', { status: 'success', value: 3 }] :
        ['eval', { code: 'const values = await Promise.all([1,2,3].map(item => nl<number>`Return one.`(item))); return values.reduce((sum, value) => sum + value, 0);' }];
      const message = { role: 'assistant', content: '', tool_calls: [{ id: 'paired', type: 'function',
        function: { name: call[0], arguments: JSON.stringify(call[1]) } }] };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const item = { index: 0, record: record('paired-plans') };
    item.record.semantics.expected = 3;
    const options = { ...config(dir), workers: 1, endpoint: `http://127.0.0.1:${server.address().port}`,
      systemPrompt: defaultSystemPrompt, modelConcurrency: 1, executionPlans: true, maxModelRequests: 6,
      transportRetries: 0, toolSurfaceSha256: await defaultToolSurfaceHash() };
    const stopped = await collectBatch([item], options, nativeJobRunner(options));
    assert.deepEqual(stopped.missing, [0]);
    assert.deepEqual(kinds, ['plan','action','plan','action','plan','action']);
    const partial = JSON.parse(await readFile(join(options.jobs, `${jobKey(item)}.partial.json`), 'utf8'));
    assert.equal(partial.turns.length, 3, 'root plus two finished child actions survive exhaustion');
    assert.equal(partial.turns.filter(t => t.response.calls[0][0] === 'return_result').length, 2);
    kinds.length = 0;
    const resumed = await collectBatch([item], options, nativeJobRunner(options));
    assert.equal(resumed.completed, 1, 'saved child decisions resume without spending the budget again');
    assert.deepEqual(kinds, ['plan','action','plan','action']);
    const row = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(row.outcome.accepted, true);
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('retry waits are durable, abortable, and removed without producing a training result', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-backoff-'));
  const options = { ...config(dir), workers: 1, transportRetries: 1, retryDelayMs: 10000 };
  const item = { index: 0, record: record('abort-backoff') }, controller = new AbortController();
  let calls = 0;
  const collecting = collectBatch([item], options, async () => {
    calls++;
    throw new Error('rate limit');
  }, controller.signal);
  const path = join(options.jobs, `${jobKey(item)}.retry.json`);
  let event;
  try {
    for (let i = 0; i < 200; i++) {
      try { event = JSON.parse(await readFile(path, 'utf8')); break; }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
      await new Promise(resolve => setTimeout(resolve, 5));
    }
    assert.equal(event.reason, 'rate_limit');
    assert.equal(event.attempt, 1);
    assert.ok(event.until > Date.now());
  } finally { controller.abort(); }
  assert.deepEqual((await collecting).missing, [0]);
  assert.equal(calls, 1);
  await assert.rejects(readFile(path), /ENOENT/);
  assert.equal(await readFile(options.output, 'utf8'), '');
});

test('a provider finish_reason error gets one bounded retry and its exact reason survives exhaustion', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-provider-finish-error-'));
  const options = { ...config(dir), workers: 1, caseEventsFile: join(dir, 'case-events.jsonl'),
    transportRetries: 1, retryDelayMs: 0 };
  const item = { index: 0, record: record('provider-finish-error') };
  let calls = 0;
  const result = await collectBatch([item], options, async () => {
    calls++;
    throw new Error('Provider finish_reason: error');
  });
  assert.deepEqual(result.missing, [0]);
  assert.equal(calls, 2, 'retry policy allows exactly one retry when configured for one');
  const failure = JSON.parse(await readFile(join(options.jobs, '000000.error.json'), 'utf8'));
  assert.equal(failure.provider_finish_reason, 'error');
  assert.equal(failure.error, 'Error: Provider finish_reason: error');
  const events = (await readFile(options.caseEventsFile, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(events.filter(event => event.event === 'case_retry').map(event => event.provider_finish_reason), ['error']);
  await assert.rejects(readFile(join(options.jobs, `${jobKey(item)}.result.json`)), /ENOENT/);
});

test('external execution adapters are caller supplied and identified in provenance',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'teacher-adapter-'));
 const item={index:0,record:record('external-fixture')};
 let received;
 const options={...config(dir),endpoint:'http://127.0.0.1:1',workers:1,execution:{identity:'application-fixture/1',run:async(program)=>{
  received=program;return {outcome:{accepted:true,kind:'done',value:1},trace:[]};
 }}};
 assert.equal(expectedProvenance(item.record,options).execution_adapter,'application-fixture/1');
 assert.equal(expectedProvenance(item.record,config(dir)).execution_adapter,undefined);
 await collectBatch([item],options,nativeJobRunner(options));
 assert.equal(received.id,item.record.id);
 const result=JSON.parse((await readFile(options.output,'utf8')).trim());
 assert.equal(result.outcome.accepted,true);
 assert.equal(result.provenance.execution_adapter,'application-fixture/1');
});
