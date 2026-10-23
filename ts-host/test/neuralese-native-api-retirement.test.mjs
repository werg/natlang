import assert from 'node:assert/strict';
import test from 'node:test';
import { HttpNeuraleseStore, neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';
import { buildStandardLibrary } from '../dist/neuralese/combinators.js';

test('native store has no flat put/get methods or projected restore path', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (input, init = {}) => {
    requests.push({ url: String(input), method: init.method ?? 'GET' });
    return new Response('not found', { status: 404 });
  });
  const remote = new HttpNeuraleseStore('http://native-fixture');
  assert.equal('put' in remote, false);
  assert.equal('get' in remote, false);
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

test('native chat transport rejects flat vector content before the request reaches the server', async t => {
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
    /does not accept flat vector blocks/);
  assert.deepEqual(requests, ['http://native-fixture/v1/neuralese/info']);
});
