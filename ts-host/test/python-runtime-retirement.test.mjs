import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const read = path => readFile(new URL(`../../${path}`, import.meta.url), 'utf8');

test('active teacher launchers use the stable Node collector entrypoint', async () => {
  const [pipeline, playground] = await Promise.all([
    read('scripts/run_teacher_generation.sh'),
    read('ts-host/scripts/playground-jobs.mjs'),
  ]);
  assert.match(pipeline, /node['"]?,?\s+(?:args:\s*\[)?['"]?ts-host\/scripts\/teacher-collector\.mjs/);
  assert.match(playground, /node['"]?,?\s+(?:args:\s*\[)?['"]?ts-host\/scripts\/playground-teacher\.mjs/);
  for (const source of [pipeline, playground]) {
    assert.doesNotMatch(source, /scripts\/collect_(?:scenario_teacher|teacher_batch)\.py/);
  }
});

test('legacy Python teacher CLIs refuse scope-eval-v1 collection', async () => {
  const [single, batch, guard] = await Promise.all([
    read('scripts/collect_scenario_teacher.py'),
    read('scripts/collect_teacher_batch.py'),
    read('scripts/python_runtime_retirement.py'),
  ]);
  assert.match(single, /refuse_legacy_scope_eval\("scripts\/collect_scenario_teacher\.py"\)/);
  assert.match(batch, /refuse_legacy_scope_eval\("scripts\/collect_teacher_batch\.py"\)/);
  assert.match(guard, /node ts-host\/scripts\/teacher-collector\.mjs instead/);
});
