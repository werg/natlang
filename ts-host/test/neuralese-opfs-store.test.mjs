import assert from 'node:assert/strict';
import test from 'node:test';
import { OpfsNeuraleseStore, openBrowserNeuraleseStore } from '../dist/browser/neuralese-opfs-store.js';
import { MemoryNeuraleseStore, encodeNativeState } from '../dist/native/neuralese-store.js';
import { decodeNz } from '../dist/native/nz-file.js';
import { fakeOpfs } from './fixtures/fake-opfs.mjs';
import { nativeStateFixture } from './fixtures/native-state.mjs';

test('OPFS stores exact native state across a reopen and ignores historical flat IDs', async () => {
  const { root, storage } = fakeOpfs();
  const source = await OpfsNeuraleseStore.open({ storage });
  const artifact = await source.putNativeState(nativeStateFixture());
  const reopened = await OpfsNeuraleseStore.open({ storage, residentBytes: 0 });
  assert.deepEqual([...encodeNativeState(await reopened.getNativeState(artifact.id))], [...encodeNativeState(artifact)]);
  assert.equal(await reopened.hasNativeState(artifact.id), true);
  assert.deepEqual(reopened.ids(), [artifact.id]);
  assert.ok(root.directories.get('natlang-native-artifacts').files.has(artifact.id));

  const oldId = 'nz1_' + 'b'.repeat(52);
  root.directories.get('natlang-native-artifacts').files.set(oldId, new Uint8Array(12));
  const third = await OpfsNeuraleseStore.open({ storage });
  assert.deepEqual(third.ids(), [artifact.id]);
  assert.deepEqual(third.unreadable, []);
  await assert.rejects(() => third.get(oldId), /neuralese-projected-api-retired/);
});

test('typed native artifacts round-trip through the browser .nz archive', async () => {
  const { storage } = fakeOpfs();
  const source = await OpfsNeuraleseStore.open({ storage, directory: 'source' });
  const artifact = await source.putNativeState(nativeStateFixture());
  const bytes = await source.exportNz([artifact.id]);
  const { blocks, nativeStates, nativeTensors } = decodeNz(bytes);
  assert.deepEqual([...blocks.keys()], []);
  assert.deepEqual([...nativeStates.keys()], [artifact.id]);
  assert.deepEqual([...nativeTensors.keys()], []);
  const target = await OpfsNeuraleseStore.open({ storage, directory: 'target' });
  assert.deepEqual(await target.importNz(bytes), [artifact.id]);
  assert.deepEqual([...encodeNativeState(await target.getNativeState(artifact.id))], [...encodeNativeState(artifact)]);
});

test('browser fallback remains a native-capable in-memory archive', async () => {
  assert.ok(await openBrowserNeuraleseStore({ storage: null }) instanceof MemoryNeuraleseStore);
  const { storage } = fakeOpfs();
  assert.ok(await openBrowserNeuraleseStore({ storage }) instanceof OpfsNeuraleseStore);
  const refusing = { async getDirectory() { throw new DOMException('denied', 'SecurityError'); } };
  assert.ok(await openBrowserNeuraleseStore({ storage: refusing }) instanceof MemoryNeuraleseStore);
});
