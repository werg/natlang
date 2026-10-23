import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';
import { encodeNativeState, MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { withNativeStates } from '../dist/neuralese/learning.js';
import { neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';
import { ModelTurnRecorder, setRecorderSource } from '../dist/neuralese/recording.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';

async function fixtureServer(state, identity = state.metadata) {
  let requestBody;
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    const send = (status, value) => {
      response.writeHead(status, { 'content-type': 'application/json' });
      response.end(JSON.stringify(value));
    };
    if (request.url === '/v1/neuralese/info') return send(200, { dialects: ['native-state/fixture'], stream: false,
      capabilities: ['chat'], native_state: { enabled: true, architecture: identity.architecture,
        checkpoint_sha256: identity.checkpoint_sha256, tokenizer_fingerprint: identity.tokenizer_fingerprint,
        embedding_weight_sha256: identity.embedding_weight_sha256, consumption_modes: ['exact-native-state'] } });
    if (request.url.endsWith('/meta')) return send(404, { error: { code: 'not-found' } });
    if (request.method === 'GET' && request.url.includes('/native-states/')) {
      response.writeHead(200, { 'content-type': 'application/octet-stream' });
      response.end(encodeNativeState(state));
      return;
    }
    if (request.method === 'PUT' && request.url.includes('/native-states/'))
      return send(201, { id: state.id });
    if (request.url === '/v1/chat/completions') {
      requestBody = JSON.parse(body.toString());
      return send(200, { choices: [{ message: { role: 'assistant', content: 'done' }, finish_reason: 'stop' }],
        x_natlang_rl_capture: { schema: 'natlang.native-rl-sampling/1', prompt: [{ kind: 'native_state', id: state.id }],
          samples: [], native_state_imports: [{ id: state.id, mode: 'exact-native-state', observed_native_state_id: state.id }] } });
    }
    return send(404, { error: { code: 'not-found' } });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${server.address().port}`, body: () => requestBody,
    close: () => new Promise(resolve => server.close(resolve)) };
}

test('withNativeStates sends state only and retains the server import receipt', async () => {
  const store = new MemoryNeuraleseStore();
  const state = await store.putNativeState(nativeStateFixture());
  const server = await fixtureServer(state);
  try {
    const recorder = new ModelTurnRecorder();
    setRecorderSource(() => recorder);
    const driver = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'fixture', store });
    await withNativeStates(state.id, () => driver({ messages: [{ role: 'user', content: 'Answer.' }], tools: [] }));
    assert.deepEqual(server.body().messages[0], { role: 'system', content: [{ type: 'native_state', id: state.id }] });
    assert.equal(server.body().messages[1].content, 'Answer.');
    assert.deepEqual(recorder.turns[0].native.sampling.native_state_imports,
      [{ id: state.id, mode: 'exact-native-state', observed_native_state_id: state.id }]);
    assert.ok(!JSON.stringify(server.body().messages).includes('the balance is 42'), 'stored provenance text must not enter the prompt');
  } finally { setRecorderSource(() => undefined); await server.close(); }
});

test('withNativeStates rejects a different current actor before sending a prompt', async () => {
  const store = new MemoryNeuraleseStore();
  const state = await store.putNativeState(nativeStateFixture());
  const server = await fixtureServer(state, { ...state.metadata, checkpoint_sha256: 'f'.repeat(64) });
  try {
    const driver = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'fixture', store });
    await assert.rejects(() => withNativeStates(state.id,
      () => driver({ messages: [{ role: 'user', content: 'Answer.' }], tools: [] })), /neuralese-native-state-identity/);
    assert.equal(server.body(), undefined);
  } finally { await server.close(); }
});

test('withNativeStates validates stored native-state IDs', async () => {
  await assert.rejects(() => withNativeStates('not-a-state', async () => {}), /learning-native-state/);
});
