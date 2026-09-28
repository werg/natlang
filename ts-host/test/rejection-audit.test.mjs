import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';

test('rejection audit counts raw jobs once and distinguishes answers from collection errors', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'rejection-audit-')), jobs = join(dir, 'jobs'), output = join(dir, 'audit');
  await mkdir(jobs);
  const row = (id, value, accepted) => ({ task: { program_ir: { id, semantics: { expected: 1 } } },
    outcome: { status: 'done', value, accepted, oracle: { accepted } }, trajectory: [] });
  await writeFile(join(jobs, '000000-a.result.json'), JSON.stringify(row('good', 1, true)));
  await writeFile(join(jobs, '000001-b.result.json'), JSON.stringify(row('bad', 2, false)));
  await writeFile(join(jobs, '000001.error.json'), JSON.stringify({ index: 1, program_id: 'bad', error: 'stale export error' }));
  await writeFile(join(jobs, '000002.error.json'), JSON.stringify({ index: 2, error: 'Context size has been exceeded.' }));
  await writeFile(join(jobs, 'range.results.jsonl'), JSON.stringify(row('bad', 2, false)) + '\n');
  await promisify(execFile)(process.execPath, [new URL('../scripts/audit-rejections.mjs', import.meta.url).pathname,
    output, jobs, jobs]);
  const summary = JSON.parse(await readFile(join(output, 'summary.json'), 'utf8'));
  assert.equal(summary.results, 2);
  assert.equal(summary.task_rejected, 1);
  assert.equal(summary.errors, 1);
  assert.deepEqual(summary.error_messages, { context_limit: 1 });
  assert.deepEqual(summary.review_categories, { answer_review: 1 });
  const rejected = JSON.parse((await readFile(join(output, 'rejections.jsonl'), 'utf8')).trim());
  assert.equal(rejected.id, 'bad');
  assert.equal(rejected.review_category, 'answer_review');
});
