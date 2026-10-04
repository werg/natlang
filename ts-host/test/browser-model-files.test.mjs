import assert from 'node:assert/strict';
import test from 'node:test';
import { cachedModelFile } from '../dist/browser/model-files.js';

/** An in-memory stand-in for the origin-private file system: a write lands only when its stream closes. */
function fakeStorage() {
  const files = new Map();
  const directory = {
    async getDirectoryHandle() { return directory; },
    async getFileHandle(name, options = {}) {
      if (!files.has(name) && !options.create) throw new DOMException('not found', 'NotFoundError');
      return {
        async getFile() { return new File([files.get(name) ?? new Uint8Array()], name); },
        async createWritable() {
          const chunks = [];
          const stream = new WritableStream({ write(chunk) { chunks.push(chunk); }, close() { files.set(name, Buffer.concat(chunks)); } });
          stream.abort = async () => undefined;
          return stream;
        },
      };
    },
    async removeEntry(name) { files.delete(name); },
  };
  return { files, storage: { async getDirectory() { return directory; } } };
}

test('model files download once into OPFS and are opened from there; a size mismatch is refused', async () => {
  const { files, storage } = fakeStorage();
  const bytes = new Uint8Array(1000).map((_, i) => i % 251);
  let fetches = 0;
  const fetcher = async () => { fetches++; return new Response(bytes); };
  const progress = [];
  const file = { url: 'https://example.test/model.gguf', bytes: 1000, sha256: 'a'.repeat(64) };
  const first = await cachedModelFile(file, { storage, fetcher, onProgress: (received, total) => progress.push([received, total]) });
  assert.deepEqual(new Uint8Array(await first.arrayBuffer()), bytes);
  assert.equal(progress.at(-1)[0], 1000);
  const again = await cachedModelFile(file, { storage, fetcher });
  assert.equal(again.size, 1000);
  assert.equal(fetches, 1, 'the second load reads the cache');
  assert.ok(files.has('sha256-' + 'a'.repeat(64)));
  await assert.rejects(() => cachedModelFile({ ...file, sha256: 'b'.repeat(64), bytes: 999 }, { storage, fetcher }), /manifest says 999/);
  assert.ok(!files.has('sha256-' + 'b'.repeat(64)), 'a wrong-size download is removed');
  // Without OPFS the file is fetched each time.
  const plain = await cachedModelFile(file, { storage: null, fetcher });
  assert.equal(plain.size, 1000);
  assert.equal(fetches, 3);
});
