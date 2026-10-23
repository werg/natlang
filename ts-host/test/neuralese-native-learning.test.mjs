import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createLearning, learningService } from '../dist/neuralese/learning.js';
import { nativeRecordedGraphGradient } from '../dist/neuralese/native-learning-client.js';
import { activeNativeAdapters } from '../dist/neuralese/recording.js';
import { MemoryNeuraleseStore, encodeNativeTensorArtifact, decodeNativeTensorArtifact } from '../dist/native/neuralese-store.js';
import { nativeAdapterRef } from '../dist/native/neuralese.js';

const identity = { checkpoint_sha256: '1'.repeat(64), tokenizer_fingerprint: '2'.repeat(64),
  embedding_weight_sha256: '3'.repeat(64), architecture: 'fixture-native' };
const tensor = values => ({ dtype: 'F32', shape: [values.length], data: new Uint8Array(new Float32Array(values).buffer) });
const values = value => [...new Float32Array(value.data.buffer.slice(value.data.byteOffset, value.data.byteOffset + value.data.byteLength))];
async function coefficient(store, name, numbers, basisId, kind = 'adapter-coefficients') {
  const value = await store.putNativeTensorArtifact({ metadata: { schema: 'natlang.neuralese-native-tensor-artifact/1', kind,
    native_identity: identity, source_artifact_id: basisId,
    provenance: { adapter_spec: { kind: 'xs', rank: 1, dim: 4, parameter_names: [name] } },
    tensors: [{ name, dtype: 'F32', shape: [numbers.length] }] }, tensors: { [name]: tensor(numbers) } });
  return value;
}
async function basis(store, name) {
  return store.putNativeTensorArtifact({ metadata: { schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'adapter-basis',
    native_identity: identity, source_artifact_id: null,
    provenance: { adapter_spec: { kind: 'xs', rank: 1, dim: 4, parameter_names: [name] } },
    tensors: [{ name: `left.${name}`, dtype: 'F32', shape: [1] }] }, tensors: { [`left.${name}`]: tensor([1]) } });
}

test('native adapter references retain the nnt1 dependency closure through learning save and load', async () => {
  const store = new MemoryNeuraleseStore(), base = await basis(store, 'layer.q.weight');
  const adapter = await coefficient(store, 'layer.q.weight', [1, 2], base.id);
  const learning = createLearning(learningService({ endpoint: 'http://127.0.0.1:1', store }));
  const directory = mkdtempSync(join(tmpdir(), 'natlang-native-learning-'));
  const path = join(directory, 'adapter.nz');
  try {
    await learning.save(path, { actor: nativeAdapterRef(adapter.id) });
    const restored = new MemoryNeuraleseStore();
    const loaded = await createLearning(learningService({ endpoint: 'http://127.0.0.1:1', store: restored })).load(path);
    assert.equal(loaded.exports.actor.$nativeAdapter.id, adapter.id);
    assert.equal((await restored.getNativeTensorArtifact(adapter.id)).id, adapter.id);
    assert.equal((await restored.getNativeTensorArtifact(base.id)).id, base.id);
    assert.deepEqual(values((await restored.getNativeTensorArtifact(adapter.id)).tensors['layer.q.weight']), [1, 2]);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('native adapter diff, apply and compose operate on exact named coefficient tensors', async () => {
  const store = new MemoryNeuraleseStore(), base = await basis(store, 'layer.q.weight');
  const zero = await coefficient(store, 'layer.q.weight', [0, 0], base.id);
  const tuned = await coefficient(store, 'layer.q.weight', [1, -2], base.id);
  const other = await coefficient(store, 'layer.q.weight', [3, 4], base.id);
  const learning = createLearning(learningService({ endpoint: 'http://127.0.0.1:1', store }));
  const delta = await learning.deltas.diff(nativeAdapterRef(tuned.id), nativeAdapterRef(zero.id));
  assert.match(delta.$nativeAdapter.id, /^nnt1_/);
  assert.deepEqual(values((await store.getNativeTensorArtifact(delta.$nativeAdapter.id)).tensors['layer.q.weight']), [1, -2]);
  const halfway = await learning.deltas.apply(nativeAdapterRef(zero.id), delta, 0.5);
  assert.deepEqual(values((await store.getNativeTensorArtifact(halfway.$nativeAdapter.id)).tensors['layer.q.weight']), [0.5, -1]);
  const combined = await learning.deltas.compose([delta, { delta: nativeAdapterRef(other.id), scale: 2 }]);
  assert.deepEqual(values((await store.getNativeTensorArtifact(combined.$nativeAdapter.id)).tensors['layer.q.weight']), [7, 6]);
});

test('withNativeAdapters scopes and nests only archived native coefficient artifacts', async () => {
  const store = new MemoryNeuraleseStore(), base = await basis(store, 'layer.q.weight');
  const adapter = await coefficient(store, 'layer.q.weight', [0, 0], base.id);
  const learning = createLearning(learningService({ endpoint: 'http://127.0.0.1:1', store }));
  const observed = await learning.withNativeAdapters({ adapter: nativeAdapterRef(adapter.id), scale: 0.25 }, async () => {
    assert.deepEqual(activeNativeAdapters(), [{ id: adapter.id, scale: 0.25 }]);
    return learning.withNativeAdapters(nativeAdapterRef(adapter.id), () => [...activeNativeAdapters()]);
  });
  assert.deepEqual(observed, [{ id: adapter.id, scale: 0.25 }, { id: adapter.id, scale: 1 }]);
  await assert.rejects(() => learning.withNativeAdapters({ $neuralese: { type: 'Adapter', id: 'nz1_legacy' } }, async () => {}), /named native adapter references/);
});

test('recorded native gradient request carries the whole actual graph and exact nnt1 target names', async () => {
  const store = new MemoryNeuraleseStore(), base = await basis(store, 'layer.q.weight');
  const adapter = await coefficient(store, 'layer.q.weight', [0, 0], base.id);
  const gradient = decodeNativeTensorArtifact(encodeNativeTensorArtifact({ metadata: {
    schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'adapter-gradient', native_identity: identity,
    source_artifact_id: adapter.id, provenance: { test_fixture: true },
    tensors: [{ name: 'layer.q.weight', dtype: 'F32', shape: [2] }] }, tensors: { 'layer.q.weight': tensor([1, -1]) } }));
  await store.putNativeTensorArtifact({ metadata: gradient.metadata, tensors: gradient.tensors });
  let received;
  const server = createServer(async (req, res) => {
    const chunks = []; for await (const chunk of req) chunks.push(chunk);
    received = JSON.parse(Buffer.concat(chunks).toString());
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ loss: 0.75, stats: { graph_nodes: 3 }, gradient: { id: gradient.id }, native_identity: identity,
      objective: 'native-query-objective/1' }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const endpoint = `http://127.0.0.1:${server.address().port}`;
  const turn = { messages: [{ role: 'user', content: 'use the written state' }], reply: { role: 'assistant', content: 'done' }, blocks: [],
    adapters: [{ id: adapter.id, scale: 0.5 }], native: { schema: 'natlang.native-recorded-turn/1', id: 'turn-1', retry: 0,
      kind: 'generation', sampling: {}, graph: { graph_turn: 'turn-node', written_nodes: {}, read_bindings: {} } } };
  const events = [{ node: 'producer', kind: 'block_write' }, { node: 'turn-node', kind: 'model_turn' }, { node: 'consumer', kind: 'block_read' }];
  try {
    const result = await nativeRecordedGraphGradient(endpoint, store, { turns: [turn], events,
      terms: [{ kind: 'logLikelihood', native_turn_id: 'turn-1' }], target: { kind: 'adapter', id: adapter.id,
        names: ['layer.q.weight'] }, nativeIdentity: identity });
    assert.equal(result.gradient.id, gradient.id);
    assert.deepEqual(received.native_recorded_objective.turns, [turn]);
    assert.deepEqual(received.native_recorded_objective.events, events);
    assert.deepEqual(received.native_recorded_objective.target, { kind: 'adapter', id: adapter.id, names: ['layer.q.weight'] });
    assert.equal(received.target, undefined, 'target remains bound inside the native recorded-objective envelope');
  } finally { await new Promise(resolve => server.close(resolve)); }
});
