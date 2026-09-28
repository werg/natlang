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
  assert.ok(started[0] < 30 && started[1] >= 40 && started[2] >= 80, `start times ${started}`);
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
