import assert from 'node:assert/strict';
import { test } from 'node:test';
import { coerce } from '../dist/native/values.js';
import { TypeEnv, parseType } from '../dist/native/types.js';
import { MemoryNeuraleseStore, encodeNativeTensorArtifact } from '../dist/native/neuralese-store.js';
import { nativeAdapterRef, nativeTensorRef, neuraleseRef } from '../dist/native/neuralese.js';
import { decodeNz, encodeNz, loadNz, loadNzSync, saveNz } from '../dist/native/nz-file.js';

const identity = { checkpoint_sha256: '1'.repeat(64), tokenizer_fingerprint: '2'.repeat(64),
  embedding_weight_sha256: '3'.repeat(64), architecture: 'native-tensor-ref-test' };
const tensor = values => ({ dtype: 'F32', shape: [values.length], data: new Uint8Array(new Float32Array(values).buffer) });
async function put(store, kind, source, provenance, name, values) {
  return store.putNativeTensorArtifact({ metadata: { schema: 'natlang.neuralese-native-tensor-artifact/1', kind,
    native_identity: identity, source_artifact_id: source, provenance,
    tensors: [{ name, dtype: 'F32', shape: [values.length] }] }, tensors: { [name]: tensor(values) } });
}

test('NativeTensor is a typed opaque tensor reference, distinct from adapter actor references', async () => {
  const store = new MemoryNeuraleseStore();
  const basis = await put(store, 'adapter-basis', null, { test: true }, 'left.layer', [1]);
  const adapter = await put(store, 'adapter-coefficients', basis.id, { test: true }, 'layer.weight', [0.5]);
  const state = await put(store, 'optimizer-state', adapter.id,
    { scope: 'natlang.native-adapter/1', step: 2 }, 'adam.exp_avg.layer.weight', [0.25]);
  const type = parseType('NativeTensor');
  const ref = nativeTensorRef(state.id, 'optimizer-state');
  assert.equal(coerce(ref, type, new TypeEnv()), ref);
  assert.throws(() => coerce(nativeAdapterRef(adapter.id), type, new TypeEnv()), /expected NativeTensor/);
  assert.throws(() => coerce(ref, parseType('NativeAdapter'), new TypeEnv()), /expected NativeAdapter/);
  assert.throws(() => coerce({ $nativeTensor: { id: state.id, kind: 'adapter-coefficients' } }, type, new TypeEnv()),
    /expected NativeTensor/);
  assert.throws(() => neuraleseRef('Neuralese<unknown>', state.id), /not a Neuralese block ID/);
  assert.throws(() => nativeTensorRef('nzs1_' + 'a'.repeat(52), 'optimizer-state'), /nnt1/);
});

test('portable NativeTensor refs restore the exact optimizer ID and full executable dependency closure', async () => {
  const source = new MemoryNeuraleseStore();
  const basis = await put(source, 'adapter-basis', null, { test: true }, 'left.layer', [1]);
  const adapter = await put(source, 'adapter-coefficients', basis.id, { test: true }, 'layer.weight', [0.5]);
  const state = await put(source, 'optimizer-state', adapter.id,
    { scope: 'natlang.native-adapter/1', step: 2 }, 'adam.exp_avg.layer.weight', [0.25]);
  const optimizerRef = nativeTensorRef(state.id, 'optimizer-state');
  const bytes = await saveNz({ optimizer: { type: 'NativeTensor', value: optimizerRef },
    history: { type: '{ current: NativeTensor; previous: NativeTensor[] }',
      value: { current: optimizerRef, previous: [optimizerRef] } } },
    { store: source, dialect: 'native-tensor/optimizer-state' });
  const portable = new MemoryNeuraleseStore();
  const loaded = await loadNz(bytes, { store: portable });
  assert.deepEqual(loaded.exports.optimizer, nativeTensorRef(state.id, 'optimizer-state'));
  assert.deepEqual(loaded.exports.history, { current: optimizerRef, previous: [optimizerRef] });
  assert.deepEqual(new Set(loaded.nativeTensors.keys()), new Set([basis.id, adapter.id, state.id]));
  assert.equal((await portable.getNativeTensorArtifact(state.id)).metadata.kind, 'optimizer-state');

  await portable.pin(state.id);
  const removed = await portable.collect(new Set());
  assert.deepEqual(removed, []);
  for (const id of [state.id, adapter.id, basis.id]) assert.equal(await portable.has(id), true);
  await portable.unpin(state.id);
  await portable.collect(new Set());
  for (const id of [state.id, adapter.id, basis.id]) assert.equal(await portable.has(id), false);
});

test('archive save and restore reject a reference whose declared tensor kind was changed', async () => {
  const store = new MemoryNeuraleseStore();
  const state = await put(store, 'optimizer-state', null, { step: 1 }, 'adam.step', [1]);
  const wrongKindBytes = await saveNz({ optimizer: { type: 'NativeTensor',
    value: nativeTensorRef(state.id, 'parameter-update') } }, { store, dialect: 'native-tensor/optimizer-state' })
    .catch(error => { assert.match(String(error), /kind does not match/); return undefined; });
  assert.equal(wrongKindBytes, undefined);

  const bytes = await saveNz({ optimizer: { type: 'NativeTensor',
    value: nativeTensorRef(state.id, 'optimizer-state') } }, { store, dialect: 'native-tensor/optimizer-state' });
  const decoded = decodeNz(bytes);
  const header = structuredClone(decoded.header);
  header.exports.optimizer.value.$nativeTensor.kind = 'parameter-update';
  const artifacts = new Map([...decoded.nativeTensors].map(([id, artifact]) => [id, encodeNativeTensorArtifact({
    metadata: artifact.metadata, tensors: artifact.tensors, canonicalMetadata: artifact.canonicalMetadata,
  })]));
  assert.throws(() => loadNzSync(encodeNz(header, artifacts)), /kind does not match its archived artifact/);
});
