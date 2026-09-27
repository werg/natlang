import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

test('held-out scorer counts missing rows and detects source overlap', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'natlang-score-'));
  const ir = join(dir, 'test.jsonl'), results = join(dir, 'results.jsonl');
  const train = join(dir, 'train.jsonl'), out = join(dir, 'score.json');
  const one = { id: 'one', source_groups: ['data:a'], curriculum: { family: 'folder_find' },
    semantics: { oracle: { level: 'normalized' } } };
  const two = { id: 'two', source_groups: ['data:b'], curriculum: { family: 'folder_find' },
    semantics: { oracle: 'exact' } };
  await writeFile(ir, [one, two].map(JSON.stringify).join('\n') + '\n');
  await writeFile(results, JSON.stringify({ task: { program_ir: one }, outcome: { status: 'done', accepted: true } }) + '\n');
  await writeFile(train, JSON.stringify({ id: 'old', source_groups: ['data:b'] }) + '\n');
  const run = spawnSync(process.execPath, ['scripts/inline-curriculum/score.mjs', '--ir', ir,
    '--results', results, '--train', train, '--out', out], { cwd: new URL('../', import.meta.url) });
  assert.equal(run.status, 2);
  const score = JSON.parse(await readFile(out, 'utf8'));
  assert.deepEqual(score.missing_ids, ['two']);
  assert.deepEqual(score.source_leakage, ['data:b']);
  assert.equal(score.by_family.folder_find.accepted, 1);
  assert.equal(score.by_oracle.normalized.completed, 1);
  assert.equal(score.by_status.missing, 1);
  await writeFile(results, JSON.stringify({ task: { program_ir: { ...one, source_groups: ['data:changed'] } },
    outcome: { status: 'done', accepted: true } }) + '\n');
  const stale = spawnSync(process.execPath, ['scripts/inline-curriculum/score.mjs', '--ir', ir,
    '--results', results, '--out', out], { cwd: new URL('../', import.meta.url) });
  assert.equal(stale.status, 1);
  assert.match(stale.stderr.toString(), /does not match the held-out program IR/);
});
