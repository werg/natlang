/**
 * The Node block archive (src/neuralese/node-block-store.ts): content-addressed safetensors files in a directory,
 * metadata indexed at open, payloads verified on read and resident within a bound. Blocks outlive the process: a
 * reopened store restores what a server lost.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { FileNeuraleseStore } from '../dist/neuralese/node-block-store.js';
import { decodeBlockBody, withRestoredBlocks } from '../dist/model/neuralese-server.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';

const DIALECT = 'nd:test@1';
const block = (seed, length = 2, width = 4, extra = {}) => ({ dialect: DIALECT, length, width, dtype: 'f32',
  data: new Uint8Array(new Float32Array(length * width).map((_, i) => seed + i / 10).buffer), ...extra });
const directory = () => join(mkdtempSync(join(tmpdir(), 'nz-store-')), 'blocks');

test('put, get, peek and meta; one safetensors file per block; equal bytes are one block that learns a type', async () => {
  const dir = directory();
  const store = await FileNeuraleseStore.open(dir);
  const meta = await store.put(block(1, 2, 4, { producer: { kind: 'test' } }));
  assert.match(meta.id, /^nz1_[a-z2-7]{52}$/);
  assert.deepEqual(readdirSync(dir), [meta.id], 'one file named by the ID, no temporary file left');
  const body = decodeBlockBody(new Uint8Array(readFileSync(join(dir, meta.id))));
  assert.equal(body.meta.id, meta.id);
  assert.deepEqual(store.peek(meta.id), meta, 'peek is synchronous');
  assert.equal(store.peek('nz1_' + 'a'.repeat(52)), undefined);
  assert.deepEqual([...(await store.get(meta.id)).data], [...block(1).data]);
  const again = await store.put(block(1, 2, 4, { type: 'Neuralese<string>', producer: { kind: 'other' } }));
  assert.equal(again.id, meta.id);
  assert.equal(again.type, 'Neuralese<string>');
  assert.deepEqual(again.producer, { kind: 'test' }, 'the first producer stays');
  assert.equal((await FileNeuraleseStore.open(dir)).peek(meta.id).type, 'Neuralese<string>', 'the learned type is on disk');
  await assert.rejects(() => store.put({ ...block(2), data: new Uint8Array(3) }), /bytes/);
});

test('a reopened store indexes every block without reading payloads; collect deletes unpinned, unreferenced files', async () => {
  const dir = directory();
  const first = await FileNeuraleseStore.open(dir);
  const ids = [];
  for (let seed = 0; seed < 3; seed++) ids.push((await first.put(block(seed))).id);
  writeFileSync(join(dir, `${ids[0]}.123.x.tmp`), new Uint8Array(5));
  writeFileSync(join(dir, 'nz1_' + 'b'.repeat(52)), new Uint8Array(3));
  const second = await FileNeuraleseStore.open(dir);
  assert.deepEqual(second.ids().sort(), [...ids].sort());
  assert.deepEqual(second.unreadable, ['nz1_' + 'b'.repeat(52)]);
  assert.equal(second.residentInUse, 0, 'no payload read at open');
  assert.equal(second.peek(ids[1]).length, 2);
  await second.pin(ids[0]);
  assert.deepEqual(await second.collect(new Set([ids[1]])), [ids[2]]);
  assert.equal(await second.has(ids[2]), false);
  assert.deepEqual((await FileNeuraleseStore.open(dir)).ids().sort(), [ids[0], ids[1]].sort());
  await assert.rejects(() => second.pin(ids[2]), /neuralese-unknown-block/);
});

test('a block whose bytes do not hash to its name is removed and reported', async () => {
  const dir = directory();
  const store = await FileNeuraleseStore.open(dir);
  const a = await store.put(block(1)), b = await store.put(block(2));
  // b's bytes under a's name: the header names b, so the reopened index skips it; overwrite the payload instead.
  const bytes = new Uint8Array(readFileSync(join(dir, a.id)));
  bytes[bytes.length - 1] ^= 0xff;
  writeFileSync(join(dir, a.id), bytes);
  const reopened = await FileNeuraleseStore.open(dir);
  assert.ok(reopened.peek(a.id), 'the header still reads');
  await assert.rejects(() => reopened.get(a.id), error => error.code === 'neuralese-block-integrity' && error.message.includes(a.id));
  assert.equal(await reopened.has(a.id), false);
  assert.deepEqual(readdirSync(dir), [b.id]);
});

test('note makes a block held elsewhere peekable only; resident payloads stay within the bound', async () => {
  const store = await FileNeuraleseStore.open(directory(), { residentBytes: 70 });
  const elsewhere = { id: 'nz1_' + 'c'.repeat(52), dialect: DIALECT, length: 9, width: 4, dtype: 'f32' };
  store.note(elsewhere);
  assert.equal(store.peek(elsewhere.id).length, 9);
  assert.equal(await store.has(elsewhere.id), false);
  await store.collect(new Set());
  assert.equal(store.peek(elsewhere.id), undefined, 'collect forgets unreferenced notes');
  const a = await store.put(block(1)), b = await store.put(block(2)), c = await store.put(block(3)); // 32 bytes each
  assert.ok(store.residentInUse <= 70, String(store.residentInUse));
  for (const id of [a.id, b.id, c.id]) assert.ok(await store.get(id));
  assert.ok(store.residentInUse <= 70);
});

test('a restart of the host and the server: the reopened archive restores the block the server lost', async () => {
  const dir = directory();
  const id = (await (await FileNeuraleseStore.open(dir)).put(block(7))).id;
  const server = new MemoryNeuraleseStore(); // the restarted server: empty
  const archive = await FileNeuraleseStore.open(dir); // the restarted host
  const result = await withRestoredBlocks(server, archive, [id], async () => (await server.has(id)) ? 'read' : 'lost');
  assert.equal(result, 'read');
  assert.deepEqual([...(await server.get(id)).data], [...block(7).data]);
});
