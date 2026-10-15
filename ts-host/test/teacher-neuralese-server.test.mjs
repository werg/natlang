import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';
const dist = process.env.NATLANG_TS_HOST_TEST_DIST;
const load = name => dist ? import(pathToFileURL(join(dist, name)).href) : import(new URL(`../dist/${name}`, import.meta.url));
const [{ definitionProject }, { nativeJobRunner, expectedProvenance, defaultToolSurfaceHash },
  { MemoryNeuraleseStore }, { encodeBlockBody }] = await Promise.all([
  load('teacher/program.js'), load('teacher/collector.js'), load('native/neuralese-store.js'), load('model/neuralese-server.js')]);

test('collector carries real server blocks across writer and reader children with exact graph edges', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'native-neuralese-collector-'));
  const store = new MemoryNeuraleseStore();
  const data = new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer);
  const block = await store.put({ dialect: 'nd:test@1', length: 2, width: 2, dtype: 'f32', data });
  const rootCode = 'const seed: Neuralese<string> = await nl<Neuralese<string>>`Write a note.`();\n' +
    'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the note and answer ok.`;\n' +
    'return await reader(seed);';
  const requests = [];
  const server = createServer(async (request, response) => {
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
      maxTurns: 8, collectionRole: 'student', neuraleseServer: true };
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
    await rm(directory, { recursive: true, force: true });
  }
});
