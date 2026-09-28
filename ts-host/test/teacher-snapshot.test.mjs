import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { snapshotJobs } from '../../scripts/snapshot_teacher_jobs.mjs';

test('live snapshot includes only completed results and preserves last export on corrupt input', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-snapshot-'));
  try {
    const output = join(dir, 'export.jsonl');
    const row = { task: { program_ir: { id: 'case' } }, outcome: { accepted: true } };
    await writeFile(join(dir, '000.result.json'), JSON.stringify(row));
    await writeFile(join(dir, '001.result.json.tmp-123'), '{');
    await writeFile(join(dir, '001.trace.jsonl'), '{');
    assert.equal(await snapshotJobs(dir, output), 1);
    const prior = await readFile(output, 'utf8');
    assert.deepEqual(JSON.parse(prior), row);
    await writeFile(join(dir, '001.result.json'), '{');
    await assert.rejects(snapshotJobs(dir, output));
    assert.equal(await readFile(output, 'utf8'), prior);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('atomic export streams chunks and preserves prior output when its producer fails', async () => {
  const { writeAtomic } = await import('../dist/teacher/collector.js');
  const { readdir } = await import('node:fs/promises');
  const dir = await mkdtemp(join(tmpdir(), 'teacher-stream-'));
  try {
    const output = join(dir, 'export.jsonl');
    async function* chunks() { yield 'first\n'; yield 'second\n'; }
    await writeAtomic(output, chunks());
    assert.equal(await readFile(output, 'utf8'), 'first\nsecond\n');
    async function* broken() { yield 'partial\n'; throw new Error('source failed'); }
    await assert.rejects(writeAtomic(output, broken()), /source failed/);
    assert.equal(await readFile(output, 'utf8'), 'first\nsecond\n');
    assert.deepEqual(await readdir(dir), ['export.jsonl']);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
