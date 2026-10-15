import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeNativeState, encodeNativeState, nativeStateContentId } from '../dist/native/neuralese-store.js';

const bytes = hex => Uint8Array.from(hex.match(/../g), value => Number.parseInt(value, 16));
const metadata = { schema: 'neuralese.native-state/1', text: 'hello', architecture: 'fixture',
  tokenizer_fingerprint: '1'.repeat(64), checkpoint_sha256: '2'.repeat(64), embedding_weight_sha256: '3'.repeat(64),
  layer_kinds: ['attention', 'finite_convolution'] };
const tensors = {
  token_ids: { dtype: 'I64', shape: [2], data: bytes('0b000000000000000c00000000000000') },
  positions: { dtype: 'I64', shape: [2], data: bytes('07000000000000000800000000000000') },
  embeddings: { dtype: 'BF16', shape: [2, 2], data: bytes('cd3d4d3e9a3ecd3e') },
  'layers.0.raw_k': { dtype: 'F16', shape: [2, 1, 2], data: bytes('003c003c003c003c') },
  'layers.0.v': { dtype: 'F16', shape: [2, 1, 2], data: bytes('0000000000000000') },
  'layers.1.gated_inputs': { dtype: 'F32', shape: [2, 3], data: bytes('0000803f0000803f0000803f0000803f0000803f0000803f') },
};

test('native-state identity and safetensors wire match Python', () => {
  const id = 'nzs1_zqbj3ixi4kqxyhv5ofcpuhabvqd76bwigo4z6wq42ujoq6lbpewq';
  assert.equal(nativeStateContentId(metadata, tensors), id);
  const wire = encodeNativeState({ metadata, tensors });
  const decoded = decodeNativeState(wire);
  assert.equal(decoded.id, id);
  assert.deepEqual([...decoded.tensors.embeddings.data], [...tensors.embeddings.data]);
  assert.deepEqual([...encodeNativeState({ metadata: decoded.metadata, tensors: decoded.tensors })], [...wire]);
  wire[wire.length - 1] ^= 1;
  assert.throws(() => decodeNativeState(wire), /content ID mismatch/);
});
