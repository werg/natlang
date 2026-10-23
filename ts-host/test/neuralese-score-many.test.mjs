/** `scoreMany` of the Neuralese server driver against a fake server (plans/BATCHED_EXECUTION.md, contract). */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';

async function fake(handler) {
  const posts = [];
  const instance = createServer((incoming, response) => {
    const chunks = [];
    incoming.on('data', chunk => chunks.push(chunk));
    incoming.on('end', () => {
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : null;
      posts.push({ path: incoming.url, body });
      handler(incoming.url, body, response);
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  return { posts, endpoint: `http://127.0.0.1:${instance.address().port}`,
    close: () => new Promise(resolve => instance.close(resolve)) };
}
const json = (response, status, body) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };
const user = text => [{ role: 'user', content: text }];

test('decisions arriving together are one POST /v1/natlang/score; prompts are adjacent; an item error stays with its caller', async () => {
  const server = await fake((path, body, response) => {
    if (path !== '/v1/natlang/score') return json(response, 404, {});
    json(response, 200, { results: body.items.map(item => item.continuations[0] === 'boom' ? { error: 'too long' } :
      { log_probs: item.continuations.map((_, index) => -index), tokens: item.continuations.map(() => 1) }) });
  });
  try {
    const { decide } = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'm' });
    const results = await Promise.allSettled([
      decide({ messages: user('a'), options: ['x', 'y'] }), decide({ messages: user('b'), options: ['boom'] }),
      decide({ messages: user('a'), options: ['z'] })]);
    const scores = server.posts.filter(post => post.path === '/v1/natlang/score');
    assert.equal(scores.length, 1);
    assert.deepEqual(server.posts.filter(post => post.path === '/v1/neuralese/decide'), []);
    assert.deepEqual(scores[0].body.items.map(item => item.messages[0].content), ['a', 'a', 'b']);
    assert.deepEqual(results[0].value.log_probs, [0, -1]);
    assert.deepEqual(results[2].value.log_probs, [0]);
    assert.equal(results[1].status, 'rejected');
    assert.match(String(results[1].reason), /too long/);
  } finally { await server.close(); }
});

test('a server without the score endpoint is asked per item through /v1/neuralese/decide, and not asked again', async () => {
  const server = await fake((path, body, response) => path === '/v1/neuralese/decide'
    ? json(response, 200, { log_probs: body.options.map(() => -2), tokens: body.options.map(() => 1) }) : json(response, 404, {}));
  try {
    const { decide } = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'm' });
    const run = () => Promise.all([decide({ messages: user('a'), options: ['x'] }), decide({ messages: user('b'), options: ['y', 'z'] })]);
    const first = await run();
    assert.deepEqual(first[1].log_probs, [-2, -2]);
    await run();
    assert.equal(server.posts.filter(post => post.path === '/v1/natlang/score').length, 1);
    assert.equal(server.posts.filter(post => post.path === '/v1/neuralese/decide').length, 4);
  } finally { await server.close(); }
});

test('a coalesced decision records its turn and applies its adapters in the caller\'s scope, not the flushing caller\'s', async () => {
  const { AsyncLocalStorage } = await import('node:async_hooks');
  const { setRecorderSource, setNativeAdapterSource } = await import('../dist/native/model-recording.js');
  const scope = new AsyncLocalStorage();
  setRecorderSource(() => scope.getStore()?.recorder);
  setNativeAdapterSource(() => scope.getStore()?.adapters);
  const store = new MemoryNeuraleseStore();
  const identity = { architecture: 'fixture-native', checkpoint_sha256: '1'.repeat(64),
    tokenizer_fingerprint: '2'.repeat(64), embedding_weight_sha256: '3'.repeat(64) };
  const tensor = values => ({ dtype: 'F32', shape: [values.length], data: new Uint8Array(new Float32Array(values).buffer) });
  const adapters = [];
  const artifactMetadata = new Map();
  const remember = async id => {
    const artifact = await store.getNativeTensorArtifact(id);
    artifactMetadata.set(id, { id: artifact.id, schema: artifact.metadata.schema, kind: artifact.metadata.kind,
      native_identity: artifact.metadata.native_identity, source_artifact_id: artifact.metadata.source_artifact_id,
      provenance: artifact.metadata.provenance, tensors: artifact.metadata.tensors });
  };
  for (let index = 0; index < 3; index++) {
    const name = `layer.${index}.weight`;
    const basis = await store.putNativeTensorArtifact({ metadata: { schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'adapter-basis',
      native_identity: identity, source_artifact_id: null,
      provenance: { adapter_spec: { kind: 'xs', rank: 1, dim: 4, parameter_names: [name] } },
      tensors: [{ name: `left.${name}`, dtype: 'F32', shape: [1] }] }, tensors: { [`left.${name}`]: tensor([1]) } });
    await remember(basis.id);
    const adapter = await store.putNativeTensorArtifact({ metadata: { schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'adapter-coefficients',
      native_identity: identity, source_artifact_id: basis.id,
      provenance: { adapter_spec: { kind: 'xs', rank: 1, dim: 4, parameter_names: [name] } },
      tensors: [{ name, dtype: 'F32', shape: [1] }] }, tensors: { [name]: tensor([0]) } });
    await remember(adapter.id);
    adapters.push({ id: adapter.id, scale: 1 });
  }
  const server = await fake((path, body, response) => {
    if (path.startsWith('/v1/neuralese/native-tensor-artifacts/') && path.endsWith('/meta')) {
      const id = path.split('/')[4];
      const metadata = artifactMetadata.get(id);
      return metadata ? json(response, 200, metadata) : json(response, 404, {});
    }
    if (path !== '/v1/natlang/score') return json(response, 404, {});
    json(response, 200, { results: body.items.map(item => ({ log_probs: item.continuations.map(() => -1), tokens: item.continuations.map(() => 1) })) });
  });
  try {
    const { decide } = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'm', store });
    const callers = ['a', 'b', 'c'].map((name, index) => ({ name, turns: [], adapters: [adapters[index]] }));
    await Promise.all(callers.map(caller => scope.run({ recorder: { record: turn => caller.turns.push(turn) }, adapters: caller.adapters },
      () => decide({ messages: user(caller.name), options: ['x', 'y'] }))));
    const scores = server.posts.filter(post => post.path === '/v1/natlang/score');
    assert.equal(scores.length, 1, 'the three decisions are one batch');
    for (const caller of callers) {
      assert.equal(caller.turns.length, 1, `${caller.name} records exactly its own decision turn`);
      assert.equal(caller.turns[0].messages[0].content, caller.name);
      assert.deepEqual(caller.turns[0].adapters, caller.adapters);
      const item = scores[0].body.items.find(entry => entry.messages[0].content === caller.name);
      assert.deepEqual(item.adapters, caller.adapters, `${caller.name}'s adapters only`);
    }
  } finally {
    setRecorderSource(() => undefined);
    setNativeAdapterSource(() => undefined);
    await server.close();
  }
});
