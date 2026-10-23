import assert from 'node:assert/strict';
import test from 'node:test';
import { HttpNeuraleseStore, neuraleseServerModelTurn, restoreNativeArtifacts } from '../dist/model/neuralese-server.js';
import { buildStandardLibrary } from '../dist/neuralese/combinators.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';

test('native artifact transport never requests the retired projected block API', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (input, init = {}) => {
    requests.push({ url: String(input), method: init.method ?? 'GET' });
    return new Response('not found', { status: 404 });
  });
  const remote = new HttpNeuraleseStore('http://native-fixture');
  await assert.rejects(() => remote.put({ dialect: 'legacy', length: 1, width: 1, dtype: 'f32', data: new Uint8Array(4) }),
    /neuralese-projected-api-retired/);
  await assert.rejects(() => remote.get('nz1_' + 'a'.repeat(52)), /neuralese-projected-api-retired/);
  assert.deepEqual(await restoreNativeArtifacts(remote, new MemoryNeuraleseStore(), ['nz1_' + 'a'.repeat(52)]),
    ['nz1_' + 'a'.repeat(52)]);
  assert.deepEqual(requests, []);
});

test('standard library authoring requires native text and state producers before any legacy request', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (input, init = {}) => {
    const url = String(input);
    requests.push(url);
    if (url.endsWith('/v1/neuralese/info')) return Response.json({ dialects: ['native-state/fixture'], width: 4,
      capabilities: ['chat'], native_state: { enabled: false } });
    throw new Error(`unexpected request ${init.method ?? 'GET'} ${url}`);
  });
  await assert.rejects(() => buildStandardLibrary({ endpoint: 'http://native-fixture' }), /neuralese-native-state-unavailable/);
  assert.deepEqual(requests, ['http://native-fixture/v1/neuralese/info']);
});

test('a flat block embedded in chat content is rejected before the request reaches the native server', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async input => {
    const url = String(input); requests.push(url);
    if (url.endsWith('/v1/neuralese/info')) return Response.json({ dialects: ['native-state/fixture'], stream: false,
      capabilities: ['chat'], native_state: { enabled: true, architecture: 'fixture',
        checkpoint_sha256: 'a'.repeat(64), tokenizer_fingerprint: 'b'.repeat(64),
        embedding_weight_sha256: 'c'.repeat(64), consumption_modes: ['exact-native-state'] } });
    throw new Error(`unexpected request ${url}`);
  });
  const turn = neuraleseServerModelTurn({ endpoint: 'http://native-fixture', model: 'fixture' });
  await assert.rejects(() => turn({ messages: [{ role: 'user', content: [{ type: 'neuralese', id: 'nz1_' + 'd'.repeat(52) }] }], tools: [] }),
    /neuralese-projected-api-retired/);
  assert.deepEqual(requests, ['http://native-fixture/v1/neuralese/info']);
});
