/** Native state and named-tensor custody in the persistent Node archive. */
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FileNeuraleseStore } from '../dist/neuralese/node-block-store.js';
import { restoreNativeArtifacts } from '../dist/model/neuralese-server.js';
import { MemoryNeuraleseStore, encodeNativeState } from '../dist/native/neuralese-store.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';

const directory = async () => join(await mkdtemp(join(tmpdir(), 'native-store-')), 'artifacts');

test('native state persists across process reopen with exact typed bytes', async () => {
  const dir = await directory();
  try {
    const first = await FileNeuraleseStore.open(dir, { residentBytes: 0 });
    const artifact = await first.putNativeState(nativeStateFixture());
    assert.match(artifact.id, /^nzs1_[a-z2-7]{52}$/);
    assert.deepEqual(await readdir(dir), [artifact.id]);
    const reopened = await FileNeuraleseStore.open(dir, { residentBytes: 0 });
    assert.deepEqual([...encodeNativeState(await reopened.getNativeState(artifact.id))], [...encodeNativeState(artifact)]);
    assert.equal(await reopened.hasNativeState(artifact.id), true);
    assert.deepEqual(reopened.ids(), [artifact.id]);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('native archive restores an exact state after the serving process loses it', async () => {
  const dir = await directory();
  try {
    const archive = await FileNeuraleseStore.open(dir);
    const artifact = await archive.putNativeState(nativeStateFixture());
    const serving = new MemoryNeuraleseStore();
    const absent = await restoreNativeArtifacts(serving, archive, [artifact.id]);
    assert.deepEqual(absent, []);
    assert.deepEqual([...encodeNativeState(await serving.getNativeState(artifact.id))], [...encodeNativeState(artifact)]);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('corrupted native artifact bytes are removed and reported', async () => {
  const dir = await directory();
  try {
    const store = await FileNeuraleseStore.open(dir, { residentBytes: 0 });
    const artifact = await store.putNativeState(nativeStateFixture());
    const wire = new Uint8Array(await readFile(join(dir, artifact.id)));
    wire[wire.length - 1] ^= 0xff;
    await writeFile(join(dir, artifact.id), wire);
    await assert.rejects(() => store.getNativeState(artifact.id), /neuralese-block-integrity/);
    assert.deepEqual(await readdir(dir), []);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('persistent native archive exposes only native artifact operations', async () => {
  const dir = await directory();
  try {
    const store = await FileNeuraleseStore.open(dir);
    assert.equal('put' in store, false);
    assert.equal('get' in store, false);
    assert.deepEqual(await readdir(dir), []);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
