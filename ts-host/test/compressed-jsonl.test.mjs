import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { compressedJsonl } from '../scripts/inline-curriculum/compressed-jsonl.mjs';

test('streaming replay compression preserves every row without an uncompressed journal', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'compressed-replay-'));
  try {
    const target = join(dir, 'rows.jsonl.zst.tmp');
    const writer = compressedJsonl(target);
    const rows = Array.from({ length: 1000 }, (_, i) => JSON.stringify({ i, text: 'λ <tool_call> body '.repeat(100) }) + '\n');
    for (const row of rows) await writer.write(row);
    await writer.close();
    const decoded = spawnSync('zstd', ['-q', '-d', '-c', target], { maxBuffer: 8 * 1024 ** 2 });
    assert.equal(decoded.status, 0, decoded.stderr.toString());
    assert.equal(decoded.stdout.toString(), rows.join(''));
    assert.ok((await readFile(target)).length < 100_000);
    await assert.rejects(readFile(join(dir, 'rows.jsonl')), { code: 'ENOENT' });
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('compression does not overwrite an existing artifact', { timeout: 5000 }, async () => {
  const dir = await mkdtemp(join(tmpdir(), 'compressed-exclusive-'));
  try {
    const target = join(dir, 'existing');
    await writeFile(target, 'preserve me');
    const writer = compressedJsonl(target);
    await assert.rejects(async () => { await writer.write('{}\n'); await writer.close(); }, { code: 'EEXIST' });
    await writer.abort(new Error('test cleanup'));
    assert.equal(await readFile(target, 'utf8'), 'preserve me');
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('destination I/O failure rejects the producer without hanging', { timeout: 5000 }, async () => {
  const dir = await mkdtemp(join(tmpdir(), 'compressed-failure-'));
  try {
    const parent = join(dir, 'not-a-directory');
    await writeFile(parent, 'file');
    const writer = compressedJsonl(join(parent, 'rows.zst'));
    await assert.rejects(async () => { await writer.write('{}\n'); await writer.close(); }, { code: 'ENOTDIR' });
    await writer.abort(new Error('test cleanup'));
  } finally { await rm(dir, { recursive: true, force: true }); }
});
