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

test('native-state contract rejects invalid identity, tensor inventory, positions, and non-finite state', () => {
  const clone = () => structuredClone({ metadata, tensors });
  let artifact = clone(); artifact.metadata.checkpoint_sha256 = 'bad';
  assert.throws(() => encodeNativeState(artifact), /metadata/);
  artifact = clone(); delete artifact.tensors['layers.0.v'];
  assert.throws(() => encodeNativeState(artifact), /invalid native-state tensor|attention layer/);
  artifact = clone(); artifact.tensors.positions.data = bytes('07000000000000000900000000000000');
  assert.throws(() => encodeNativeState(artifact), /positions/);
  artifact = clone(); artifact.tensors.embeddings.data = bytes('807f4d3e9a3ecd3e');
  assert.throws(() => encodeNativeState(artifact), /non-finite/);
});

test('decoder applies the same semantic checks even when a malformed body has a matching ID', () => {
  const artifact = structuredClone({ metadata, tensors });
  artifact.tensors.positions.data = bytes('07000000000000000900000000000000');
  const id = nativeStateContentId(metadata, tensors);
  const entries = {};
  let offset = 0;
  for (const name of Object.keys(artifact.tensors).sort()) {
    const tensor = artifact.tensors[name];
    entries[name] = { dtype: tensor.dtype, shape: tensor.shape, data_offsets: [offset, offset + tensor.data.length] };
    offset += tensor.data.length;
  }
  entries.__metadata__ = { 'natlang.native_state': JSON.stringify({ id, ...artifact.metadata }) };
  const encoder = new TextEncoder();
  const header = encoder.encode(JSON.stringify(Object.fromEntries(Object.entries(entries).sort(([a], [b]) => a.localeCompare(b)))));
  const aligned = new Uint8Array(Math.ceil(header.length / 8) * 8).fill(0x20); aligned.set(header);
  const payload = new Uint8Array(Object.keys(artifact.tensors).sort().reduce((n, name) => n + artifact.tensors[name].data.length, 0));
  let at = 0;
  for (const name of Object.keys(artifact.tensors).sort()) { payload.set(artifact.tensors[name].data, at); at += artifact.tensors[name].data.length; }
  const wire = new Uint8Array(8 + aligned.length + payload.length);
  new DataView(wire.buffer).setBigUint64(0, BigInt(aligned.length), true); wire.set(aligned, 8); wire.set(payload, 8 + aligned.length);
  assert.throws(() => decodeNativeState(wire), /positions/);
});
