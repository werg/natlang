import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';
const dist = process.env.NATLANG_TS_HOST_TEST_DIST;
const load = name => dist ? import(pathToFileURL(join(dist, name)).href) : import(new URL(`../dist/${name}`, import.meta.url));
const [{ definitionProject }, { nativeJobRunner, expectedProvenance, defaultToolSurfaceHash, collectBatch },
  { MemoryNeuraleseStore, encodeNativeState }, { encodeBlockBody }] = await Promise.all([
  load('teacher/program.js'), load('teacher/collector.js'), load('native/neuralese-store.js'), load('model/neuralese-server.js')]);
import { nativeStateFixture } from './fixtures/native-state.mjs';

for (const missingOnReplay of [false, true]) test(`interrupted collector hydrates exact saved native writes before replay (missing=${missingOnReplay})`, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'native-replay-custody-'));
  const remote = new MemoryNeuraleseStore();
  const block = await remote.putNativeState(nativeStateFixture());
  const rawInput = nativeStateFixture(); rawInput.metadata.text = 'raw producer capture';
  const rawBlock = await remote.putNativeState(rawInput);
  const sampling = { schema: 'natlang.native-rl-sampling/1', fixture: 'exact-retained-receipt',
    prompt: [{ kind: 'native_state', id: rawBlock.id, source_state_id: rawBlock.id }],
    writes: [{ state_id: block.id, raw_state_id: rawBlock.id }] };
  const rootCode = 'const seed: Neuralese<string> = await nl<Neuralese<string>>`Write a note.`();\n' +
    'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the note and answer ok.`;\n' +
    'return await reader(seed);';
  const requests = [], gets = [];
  const interruptedRun = new AbortController();
  let interrupted = false;
  const server = createServer(async (request, response) => {
    if (request.url === '/v1/neuralese/info') {
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ dialects: ['native-state/fixture'], stream: false, capabilities: ['chat'],
        native_state: { enabled: true, ...block.metadata, consumption_modes: ['exact-native-state'] } })); return;
    }
    if (request.url.startsWith('/v1/neuralese/native-states/')) {
      const id = request.url.split('/').at(-1); gets.push(id);
      const value = await remote.getNativeState(id);
      if (!value || missingOnReplay && interrupted) { response.writeHead(404); response.end(); return; }
      response.setHeader('content-type', 'application/octet-stream'); response.end(encodeNativeState(value)); return;
    }
    let body = ''; for await (const chunk of request) body += chunk;
    const parsed = JSON.parse(body); requests.push(parsed);
    const n = requests.length;
    if (n === 3) {
      interrupted = true; interruptedRun.abort(new Error('fixture interrupted after saved native producer'));
      request.socket.destroy(); return;
    }
    const call = n === 1 ? ['eval', { code: rootCode, finish: true }] : n === 2 ? ['eval', {
      code: [{ type: 'text', text: 'return ' }, { type: 'native_state', id: block.id }, { type: 'text', text: ';' }], finish: true,
    }] : ['return_result', { status: 'success', value: 'ok' }];
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '', tool_calls: [{
      id: `action-${n}`, type: 'function', function: { name: call[0], arguments: JSON.stringify(call[1]) },
    }] } }], ...(n === 2 ? { native_state: { blocks: [{ id: block.id }] }, x_natlang_rl_capture: sampling } : {}),
      usage: { prompt_tokens: 16, completion_tokens: 8 } }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const project = definitionProject('valueflow', { returns: 'string', instructions: 'Pass a note to the reader.' });
    const record = { version: 'natlang.program/2', id: 'retained-native-value-flow', kind: 'lambda_source', source: 'test',
      split: 'train', source_ids: ['retained-native-value-flow'], source_groups: ['test:retained-native-flow'], license: 'test',
      semantics: { root: project.root, files: project.files, inputs: {}, expected: 'ok', operation: 'exact' } };
    const config = { jobs: join(directory, 'jobs'), output: join(directory, 'out.jsonl'), workers: 1,
      modelId: 'native-replay-fixture', endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 42,
      systemPrompt: 'Test.', contextTokens: 8192, toolSurfaceSha256: await defaultToolSurfaceHash(),
      maxTurns: 8, collectionRole: 'student', neuraleseServer: true,
      transportRetries: 0, requestRetries: 0, retryDelayMs: 0 };
    await mkdir(config.jobs, { recursive: true });
    const first = await collectBatch([{ index: 0, record }], config, nativeJobRunner(config), interruptedRun.signal);
    assert.equal(interrupted, true, 'fixture really interrupted after the producer response was saved');
    assert.deepEqual(first.missing, [0], 'aborted first collector leaves its saved prefix unfinished');
    const beforeReplay = gets.length;
    const result = await collectBatch([{ index: 0, record }], config, nativeJobRunner(config));
    assert.equal(result.completed, missingOnReplay ? 0 : 1, JSON.stringify({ result, gets, beforeReplay, requests: requests.length }));
    assert.equal(requests.length, missingOnReplay ? 3 : 4, 'saved root/producer replies are never resampled');
    assert.ok(gets.slice(beforeReplay).includes(block.id), 'fresh collector archives the saved producer again');
    if (!missingOnReplay) {
      assert.ok(gets.filter(id => id === rawBlock.id).length >= 2, 'raw sampling capture roots survive replay too');
      const row = JSON.parse((await readFile(config.output, 'utf8')).trim());
      assert.equal(row.outcome.accepted, true, JSON.stringify(row.outcome));
      const write = row.outcome.execution_graph.find(node => node.kind === 'block_write' && node.block === block.id);
      const read = row.outcome.execution_graph.find(node => node.kind === 'block_read' && node.block === block.id && node.call_id !== write?.call_id);
      assert.ok(write && read?.inputs.some(input => input.node === write.node), 'replayed native producer still owns the exact consumer graph edge');
      assert.deepEqual(row.trajectory[1].model_response.rl_sampling, sampling);
      assert.equal(row.trajectory.length, 3);
    }
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  }
});

test('collector carries real server blocks across writer and reader children with exact graph edges', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'native-neuralese-collector-'));
  const store = new MemoryNeuraleseStore();
  const data = new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer);
  const block = await store.put({ dialect: 'nd:test@1', length: 2, width: 2, dtype: 'f32', data });
  const rootCode = 'const seed: Neuralese<string> = await nl<Neuralese<string>>`Write a note.`();\n' +
    'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the note and answer ok.`;\n' +
    'return await reader(seed);';
  const requests = [];
  process.env.NATLANG_RL_TEST_KEY = 'fixture-key';
  const server = createServer(async (request, response) => {
    if (request.headers.authorization !== 'Bearer fixture-key') { response.writeHead(401); response.end(); return; }
    if (request.url === '/v1/neuralese/info') {
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ dialects: ['nd:test@1'], stream: false })); return;
    }
    if (request.url === '/v1/neuralese/blocks/' + block.id) {
      response.setHeader('content-type', 'application/octet-stream');
      response.end(encodeBlockBody(block, data)); return;
    }
    let body = '';
    for await (const chunk of request) body += chunk;
    const parsed = JSON.parse(body); requests.push(parsed);
    const n = requests.length;
    const call = n === 1 ? ['eval', { code: rootCode, finish: true }] : n === 2 ? ['eval', {
      code: [{ type: 'text', text: 'return ' }, { type: 'neuralese', id: block.id }, { type: 'text', text: ';' }], finish: true,
    }] : ['return_result', { status: 'success', value: 'ok' }];
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '', tool_calls: [{
      id: `action-${n}`, type: 'function', function: { name: call[0], arguments: JSON.stringify(call[1]) },
    }] } }], ...(n === 2 ? { neuralese: { blocks: [block] } } : {}),
      usage: { prompt_tokens: 16, completion_tokens: 8 } }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const project = definitionProject('valueflow', { returns: 'string', instructions: 'Pass a note to the reader.' });
    const record = { version: 'natlang.program/2', id: 'real-value-flow', kind: 'lambda_source', source: 'test',
      split: 'train', source_ids: ['real-value-flow'], source_groups: ['test:value-flow'], license: 'test',
      semantics: { root: project.root, files: project.files, inputs: {}, expected: 'ok', operation: 'exact' } };
    const config = { jobs: join(directory, 'jobs'), output: join(directory, 'out.jsonl'), workers: 1,
      modelId: 'real-protocol-fixture', endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 42,
      systemPrompt: 'Test.', contextTokens: 8192, toolSurfaceSha256: await defaultToolSurfaceHash(),
      maxTurns: 8, collectionRole: 'student', neuraleseServer: true, apiKeyEnv: 'NATLANG_RL_TEST_KEY' };
    await mkdir(config.jobs, { recursive: true });
    const result = await nativeJobRunner(config)({ index: 0, record }, expectedProvenance(record, config));
    assert.equal(result.outcome.accepted, true, JSON.stringify(result.outcome));
    assert.equal(requests.length, 3);
    assert.ok(JSON.stringify(requests[2].messages).includes(`"type":"neuralese","id":"${block.id}"`));
    const graph = result.outcome.execution_graph;
    const write = graph.find(node => node.kind === 'block_write' && node.block === block.id);
    assert.ok(write, 'server-produced block has a writer occurrence');
    const read = graph.find(node => node.kind === 'block_read' && node.block === block.id && node.call_id !== write.call_id);
    assert.ok(read?.inputs.some(input => input.node === write.node), 'reader links exact writer occurrence');
    assert.equal(result.trajectory.filter(turn => turn.invocation_id).length, 3);
  } finally {
    await new Promise(resolve => server.close(resolve));
    delete process.env.NATLANG_RL_TEST_KEY;
    await rm(directory, { recursive: true, force: true });
  }
});
