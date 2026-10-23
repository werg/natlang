/** Native state and named-tensor custody across a serving process restart. */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { isUnknownNativeArtifactError, restoreNativeArtifacts, withRestoredNativeArtifacts } from '../dist/model/neuralese-server.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';

test('native missing-artifact errors are recognized without accepting the removed flat-block error', () => {
  assert.ok(isUnknownNativeArtifactError(Object.assign(new Error('HTTP 400'), { providerCode: 'neuralese-unknown-native-state' })));
  assert.ok(isUnknownNativeArtifactError(new Error('HTTP 400: neuralese-unknown-native-artifact')));
  assert.ok(!isUnknownNativeArtifactError(Object.assign(new Error('x'), { code: 'neuralese-unknown-block' })));
  assert.ok(!isUnknownNativeArtifactError(new Error('HTTP 500: internal')));
});

test('restoreNativeArtifacts copies an exact state and rejects IDs outside the native artifact domains', async () => {
  const archive = new MemoryNeuraleseStore(), remote = new MemoryNeuraleseStore();
  const state = await archive.putNativeState(nativeStateFixture());
  assert.deepEqual(await restoreNativeArtifacts(remote, archive, [state.id]), []);
  assert.deepEqual(await remote.getNativeState(state.id), state);
  assert.deepEqual(await restoreNativeArtifacts(remote, archive, ['nzs1_' + 'a'.repeat(52)]), ['nzs1_' + 'a'.repeat(52)]);
  await assert.rejects(restoreNativeArtifacts(remote, archive, ['nz1_' + 'b'.repeat(52)]), /native artifact restore requires an nzs1 state or nnt1 tensor ID/);
});

test('withRestoredNativeArtifacts reuploads after a serving restart and retries once', async () => {
  const archive = new MemoryNeuraleseStore(), remote = new MemoryNeuraleseStore();
  const state = await archive.putNativeState(nativeStateFixture());
  const known = new Set();
  const run = async () => (await remote.hasNativeState(state.id)) ? 'ok' : Promise.reject(new Error(`neuralese-unknown-native-state: ${state.id}`));
  assert.equal(await withRestoredNativeArtifacts(remote, archive, [state.id], run, known), 'ok');
  await remote.collect(new Set());
  let calls = 0;
  assert.equal(await withRestoredNativeArtifacts(remote, archive, [state.id], () => { calls++; return run(); }, known), 'ok');
  assert.equal(calls, 2);
  await remote.collect(new Set());
  await assert.rejects(withRestoredNativeArtifacts(remote, new MemoryNeuraleseStore(), [state.id], run),
    /neuralese-unknown-native-artifact/);
  await assert.rejects(withRestoredNativeArtifacts(remote, archive, [], () => Promise.reject(new Error('HTTP 500'))), /HTTP 500/);
});
