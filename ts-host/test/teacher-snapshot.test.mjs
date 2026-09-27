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
