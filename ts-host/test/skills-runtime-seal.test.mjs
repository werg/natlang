import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { verifyRuntimeFiles } from '../scripts/skills/verify-runtime-files.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');

test('sealed runtime verification streams every file, reports progress, and fails on content or closure gaps', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-runtime-seal-'));
  try {
    const files = {};
    for (let index = 0; index < 5; index++) {
      const path = `nested/file-${index}.txt`, bytes = index === 4 ? Buffer.alloc(160_000, 0x5a) : Buffer.from(`sealed-${index}`);
      await mkdir(join(root, 'nested'), { recursive: true });
      await writeFile(join(root, path), bytes);
      files[path] = sha(bytes);
    }
    const progress = [];
    const result = await verifyRuntimeFiles(root, files, { concurrency: 2, progressEvery: 2, onProgress: value => progress.push(value) });
    assert.equal(result.files, 5);
    assert.deepEqual(progress.map(value => value.completed), [2, 4, 5]);
    assert.ok(progress.every(value => value.total === 5 && value.elapsedMs >= 0));

    await writeFile(join(root, 'nested/file-3.txt'), 'tampered');
    await assert.rejects(() => verifyRuntimeFiles(root, files, { concurrency: 2 }), /Frozen runtime file changed: nested\/file-3\.txt/);
    await writeFile(join(root, 'nested/file-3.txt'), 'sealed-3');
    await assert.rejects(() => verifyRuntimeFiles(root, { ...files, 'nested/missing.txt': sha('missing') }), /ENOENT/);
    await assert.rejects(() => verifyRuntimeFiles(root, { '../escape': sha('escape') }), /invalid sealed runtime file entry/);
    await assert.rejects(() => verifyRuntimeFiles(root, {}), /file list is empty/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
