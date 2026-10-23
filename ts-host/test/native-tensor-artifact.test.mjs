import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { decodeNativeTensorArtifact, encodeNativeTensorArtifact, nativeTensorArtifactContentId } from '../dist/native/neuralese-store.js';
import { FileNeuraleseStore } from '../dist/neuralese/node-block-store.js';

const bytes = hex => Uint8Array.from(hex.match(/../g), byte => Number.parseInt(byte, 16));
const metadata = { schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'parameter-gradient',
  native_identity: { checkpoint_sha256: '1'.repeat(64), tokenizer_fingerprint: '2'.repeat(64),
    embedding_weight_sha256: '3'.repeat(64), architecture: 'lfm2' }, source_artifact_id: null,
  provenance: { objective: 'native-graph-token-ce/1', step: 4 },
  tensors: [{ name: 'adapter.up.weight', dtype: 'F32', shape: [2] }] };
const tensors = { 'adapter.up.weight': { dtype: 'F32', shape: [2], data: bytes('0000803f000000c0') } };

test('native tensor artifact safetensors preserves named gradient bytes and exact identity', () => {
  const id = nativeTensorArtifactContentId(metadata, tensors), wire = encodeNativeTensorArtifact({ metadata, tensors });
  const decoded = decodeNativeTensorArtifact(wire);
  assert.equal(id, 'nnt1_klptchy2xnkshs2z4n4ulvv2z24gsxgmjetrt6ipszgqpyjqkacq',
    'content address matches Python NativeTensorArtifact bytes');
  assert.match(id, /^nnt1_[a-z2-7]{52}$/);
  assert.equal(decoded.id, id);
  assert.deepEqual([...decoded.tensors['adapter.up.weight'].data], [...tensors['adapter.up.weight'].data]);
  assert.deepEqual([...encodeNativeTensorArtifact(decoded)], [...wire]);
});

test('native tensor artifacts survive the shared persistent store lifecycle without flat-vector access', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-nnt-'));
  try {
    let store = await FileNeuraleseStore.open(directory, { residentBytes: 0 });
    const saved = await store.putNativeTensorArtifact({ metadata, tensors });
    store = await FileNeuraleseStore.open(directory, { residentBytes: 0 });
    await store.pin(saved.id);
    const restored = await store.getNativeTensorArtifact(saved.id);
    assert.equal(restored.id, saved.id);
    assert.equal(await store.hasNativeState(saved.id), false);
    assert.equal(await store.hasNativeTensorArtifact(saved.id), true);
    assert.equal(store.peek(saved.id).native_tensor_artifact.kind, 'parameter-gradient');
    assert.deepEqual(await store.collect(new Set()), []);
    await store.unpin(saved.id);
    assert.deepEqual(await store.collect(new Set()), [saved.id]);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
