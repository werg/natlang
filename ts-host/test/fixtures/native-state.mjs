export function nativeStateFixture(architecture = 'fixture') {
  const f32 = values => new Uint8Array(new Float32Array(values).buffer);
  const i64 = values => new Uint8Array(new BigInt64Array(values.map(BigInt)).buffer);
  return { metadata: { schema: 'neuralese.native-state/1', text: 'the balance is 42', architecture,
    tokenizer_fingerprint: '1'.repeat(64), checkpoint_sha256: '2'.repeat(64), embedding_weight_sha256: '3'.repeat(64),
    layer_kinds: ['attention', 'finite_convolution'] }, tensors: {
      token_ids: { dtype: 'I64', shape: [2], data: i64([4, 5]) },
      positions: { dtype: 'I64', shape: [2], data: i64([8, 9]) },
      embeddings: { dtype: 'F32', shape: [2, 2], data: f32([1, 2, 3, 4]) },
      'layers.0.raw_k': { dtype: 'F32', shape: [2, 1, 2], data: f32([.1, .2, .3, .4]) },
      'layers.0.v': { dtype: 'F32', shape: [2, 1, 2], data: f32([5, 6, 7, 8]) },
      'layers.1.gated_inputs': { dtype: 'F32', shape: [2, 2], data: f32([9, 10, 11, 12]) },
  } };
}
