import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { requireNativeReaderIdentity } from '../dist/model/index.js';
import { prepareData } from '../dist/tune/pipeline.js';

const identity = { architecture: 'lfm2', checkpoint_sha256: 'a'.repeat(64), tokenizer_fingerprint: 'b'.repeat(64),
  embedding_weight_sha256: 'c'.repeat(64) };

test('native tune reader preflight requires the exact configured student checkpoint identity', async () => {
  let reader = { ...identity, consumption_modes: ['exact-native-state'] };
  const server = createServer((request, response) => {
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ dialects: ['neuralese-native-v1'], native_state: { enabled: true, ...reader } }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const endpoint = `http://127.0.0.1:${server.address().port}/v1`;
  try {
    assert.deepEqual(await requireNativeReaderIdentity(endpoint, identity), { ...identity, consumption_modes: ['exact-native-state'] });
    reader = { ...reader, checkpoint_sha256: 'd'.repeat(64) };
    await assert.rejects(requireNativeReaderIdentity(endpoint, identity), /does not match the configured student.base/);
    reader = { ...reader, enabled: false };
    await assert.rejects(requireNativeReaderIdentity(endpoint, identity), /does not match the configured student.base/);
  } finally { server.close(); }
});

test('native tune data preparation refuses to begin without a live student identity guard', async () => {
  await assert.rejects(prepareData({ settings: { backend: 'native' } }), /live student\.base identity preflight/);
  let checked = false;
  await assert.rejects(prepareData({ settings: { backend: 'native' }, assertStudentBase: async () => {
    checked = true; throw new Error('reader mismatch');
  } }), /reader mismatch/);
  assert.equal(checked, true, 'the endpoint identity check runs before data preparation or collection');
});
