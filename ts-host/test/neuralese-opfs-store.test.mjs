import assert from 'node:assert/strict';
import test from 'node:test';
import { OpfsNeuraleseStore, openBrowserNeuraleseStore } from '../dist/browser/neuralese-opfs-store.js';
import { withRestoredBlocks } from '../dist/model/neuralese-server.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { decodeNz } from '../dist/native/nz-file.js';
import { fakeOpfs } from './fixtures/fake-opfs.mjs';

const DIALECT = 'nd:test@1';
const block = (seed, length = 2, width = 4, extra = {}) => ({ dialect: DIALECT, length, width, dtype: 'f32',
  data: new Uint8Array(new Float32Array(length * width).map((_, i) => seed + i / 10).buffer), ...extra });

test('put, get, peek and meta; equal bytes are one block that learns a type', async () => {
  const { root, storage } = fakeOpfs();
  const store = await OpfsNeuraleseStore.open({ storage });
  const meta = await store.put(block(1, 2, 4, { producer: { kind: 'test' } }));
  assert.match(meta.id, /^nz1_[a-z2-7]{52}$/);
  assert.ok(root.directories.get('natlang-neuralese-blocks').files.has(meta.id), 'one file named by the ID');
  assert.deepEqual(store.peek(meta.id), meta, 'peek is synchronous');
  assert.equal(store.peek('nz1_' + 'a'.repeat(52)), undefined);
  assert.equal(await store.has(meta.id), true);
  const got = await store.get(meta.id);
  assert.deepEqual([...got.data], [...block(1).data]);
  const again = await store.put(block(1, 2, 4, { type: 'Neuralese<string>', producer: { kind: 'other' } }));
  assert.equal(again.id, meta.id);
  assert.equal(again.type, 'Neuralese<string>');
  assert.deepEqual(again.producer, { kind: 'test' }, 'the first producer stays');
  assert.equal(store.size, 1);
  // The learned type is on file: a reopened store knows it.
  const reopened = await OpfsNeuraleseStore.open({ storage });
  assert.equal(reopened.peek(meta.id).type, 'Neuralese<string>');
  await assert.rejects(() => store.put({ ...block(2), data: new Uint8Array(3) }), /bytes/);
});

test('blocks survive a reopen; metadata is indexed at open and payloads are read on demand', async () => {
  const { root, storage, stats } = fakeOpfs();
  const first = await OpfsNeuraleseStore.open({ storage });
  const ids = [];
  for (let seed = 0; seed < 3; seed++) ids.push((await first.put(block(seed))).id);
  // A swap file of an unfinished write and a stray file are skipped.
  root.directories.get('natlang-neuralese-blocks').files.set(`${ids[0]}.crswap`, new Uint8Array(5));
  root.directories.get('natlang-neuralese-blocks').files.set('nz1_' + 'b'.repeat(52), new Uint8Array(3));
  const second = await OpfsNeuraleseStore.open({ storage });
  assert.deepEqual(second.ids().sort(), [...ids].sort());
  assert.deepEqual(second.unreadable, ['nz1_' + 'b'.repeat(52)]);
  assert.equal(second.residentInUse, 0, 'no payload read at open');
  const before = stats.reads;
  assert.equal(second.peek(ids[1]).length, 2);
  assert.equal(stats.reads, before, 'peek reads nothing');
  assert.deepEqual([...(await second.get(ids[1])).data], [...block(1).data]);
  // Pins keep blocks through collection; collection deletes the files of the rest.
  await second.pin(ids[0]);
  assert.deepEqual((await second.collect(new Set([ids[1]]))), [ids[2]]);
  assert.equal(await second.has(ids[2]), false);
  assert.deepEqual((await OpfsNeuraleseStore.open({ storage })).ids().sort(), [ids[0], ids[1]].sort());
  await assert.rejects(() => second.pin(ids[2]), /neuralese-unknown-block/);
});

test('a block another tab wrote is found on a lookup miss', async () => {
  const { storage } = fakeOpfs();
  const mine = await OpfsNeuraleseStore.open({ storage });
  const theirs = await OpfsNeuraleseStore.open({ storage });
  const { id } = await theirs.put(block(7));
  assert.equal(mine.peek(id), undefined);
  assert.equal((await mine.get(id)).meta.id, id);
  assert.ok(mine.peek(id), 'indexed after the lookup');
});

test('a block whose bytes do not hash to its ID is refused and removed', async () => {
  const { root, storage } = fakeOpfs();
  const store = await OpfsNeuraleseStore.open({ storage, residentBytes: 0 });
  const { id } = await store.put(block(3));
  const files = root.directories.get('natlang-neuralese-blocks').files;
  const bytes = files.get(id).slice();
  bytes[bytes.length - 1] ^= 0xff;
  files.set(id, bytes);
  await assert.rejects(() => store.get(id), /neuralese-block-integrity/);
  assert.equal(files.has(id), false);
  assert.equal(store.peek(id), undefined);
  assert.equal(await store.get(id), undefined);
});

test('resident payloads stay within the byte budget, least recently used first out', async () => {
  const { storage, stats } = fakeOpfs();
  const size = 2 * 4 * 4; // one 2x4 f32 block
  const store = await OpfsNeuraleseStore.open({ storage, residentBytes: 2 * size });
  const ids = [];
  for (let seed = 0; seed < 3; seed++) ids.push((await store.put(block(seed))).id);
  assert.equal(store.residentInUse, 2 * size, 'the first block was evicted');
  let reads = stats.reads;
  await store.get(ids[2]); await store.get(ids[1]);
  assert.equal(stats.reads, reads, 'resident blocks are served from memory');
  await store.get(ids[0]);
  assert.equal(stats.reads, reads + 1, 'an evicted block is read from OPFS');
  // ids[2] was least recently used, so reading ids[0] evicted it.
  reads = stats.reads;
  await store.get(ids[1]);
  assert.equal(stats.reads, reads);
  await store.get(ids[2]);
  assert.equal(stats.reads, reads + 1);
  assert.ok(store.residentInUse <= 2 * size);
});

test('restore after a page reload and an engine restart uploads blocks from the reopened archive', async () => {
  const { storage } = fakeOpfs();
  const page = await OpfsNeuraleseStore.open({ storage });
  const ids = [(await page.put(block(1))).id, (await page.put(block(2))).id];
  // Reload: a new store object over the same OPFS directory, and a new engine with an empty block store.
  const archive = await OpfsNeuraleseStore.open({ storage });
  for (const engine of [new MemoryNeuraleseStore(), new MemoryNeuraleseStore()]) {
    let runs = 0;
    const result = await withRestoredBlocks(engine, archive, ids, async () => {
      runs++;
      for (const id of ids) assert.equal(await engine.has(id), true, 'on the engine before the request');
      return 'ok';
    });
    assert.equal(result, 'ok');
    assert.equal(runs, 1);
  }
  // Without the archive (an in-memory store after a reload) the blocks are reported missing.
  await assert.rejects(() => withRestoredBlocks(new MemoryNeuraleseStore(), new MemoryNeuraleseStore(), ids, async () => 'x'),
    /neither on the server nor in the runtime's store/);
});

test('.nz export and import by ID round-trip', async () => {
  const { storage } = fakeOpfs();
  const source = await OpfsNeuraleseStore.open({ storage, directory: 'a' });
  const ids = [(await source.put(block(1, 3, 4, { producer: { kind: 'write' }, truncated: true }))).id, (await source.put(block(2))).id];
  const bytes = await source.exportNz(ids);
  const { header, blocks } = decodeNz(bytes);
  assert.equal(header.dialect, DIALECT);
  assert.deepEqual(Object.keys(header.exports), []);
  assert.deepEqual([...blocks.keys()].sort(), [...ids].sort());
  const target = await OpfsNeuraleseStore.open({ storage, directory: 'b' });
  assert.deepEqual((await target.importNz(bytes)).sort(), [...ids].sort());
  for (const id of ids) {
    const [from, to] = [await source.get(id), await target.get(id)];
    assert.deepEqual([...to.data], [...from.data]);
    assert.deepEqual({ ...to.meta }, { ...from.meta });
  }
  await assert.rejects(() => source.exportNz(['nz1_' + 'c'.repeat(52)]), /neuralese-unknown-block/);
});

test('without OPFS the browser archive is in memory', async () => {
  assert.ok(await openBrowserNeuraleseStore({ storage: null }) instanceof MemoryNeuraleseStore);
  const { storage } = fakeOpfs();
  assert.ok(await openBrowserNeuraleseStore({ storage }) instanceof OpfsNeuraleseStore);
  const refusing = { async getDirectory() { throw new DOMException('denied', 'SecurityError'); } };
  assert.ok(await openBrowserNeuraleseStore({ storage: refusing }) instanceof MemoryNeuraleseStore);
});
