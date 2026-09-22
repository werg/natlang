import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { test } from 'node:test';
import { auditCoverage, buildSelection } from '../scripts/teacher-coverage.mjs';

test('native coverage selection balances and interleaves every family deterministically', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-coverage-'));
  const source = join(root, 'source.jsonl'), rows = [];
  for (const family of ['b', 'a']) for (let index = 0; index < 3; index++)
    rows.push({ version: 'natlang.program/1', id: `${family}-${index}`, kind: 'lambda_graph',
      family, split: 'train', semantics: { root: {}, inputs: {}, expected: index } });
  await writeFile(source, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
  const first = await buildSelection([source], { perFamily: 2, seed: 7 });
  const second = await buildSelection([source], { perFamily: 2, seed: 7 });
  assert.deepEqual(first, second);
  assert.equal(first.rows.length, 4);
  assert.deepEqual(first.rows.map(row => row.family), ['a', 'b', 'a', 'b']);
});

test('native coverage audit detects unregistered sources and verifies provider minimums', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-audit-'));
  await mkdir(join(root, 'codebases', 'known'), { recursive: true });
  await mkdir(join(root, 'applications'), { recursive: true });
  await writeFile(join(root, 'codebases', 'known', 'main.nl'), 'Return one.');
  await writeFile(join(root, 'applications', 'extra.mjs'), 'export {};');
  await writeFile(join(root, 'applications', 'retired.py'), '');
  const config = join(root, 'coverage.json'), studio = join(root, 'studio.jsonl'), programs = join(root, 'programs.jsonl');
  await writeFile(config, JSON.stringify({ excluded: [], targets: [{ id: 'family:f', provider: 'program_ir',
    owns: ['codebase:known'] }], minimums: { program_ir: { train: 1 } } }));
  await writeFile(studio, '');
  await writeFile(programs, JSON.stringify({ id: 'p', family: 'f', split: 'train' }) + '\n');
  const report = await auditCoverage(root, config, studio, programs);
  assert.equal(report.ready, false);
  assert.deepEqual(report.unknown, ['application:extra.mjs']);
  assert.deepEqual(report.below_minimum, []);
});

test('teacher generation snapshots native synthetic IR and Studio cases on every wave', async () => {
  const [pipeline, selector, configText] = await Promise.all([
    readFile(new URL('../../scripts/run_teacher_generation.sh', import.meta.url), 'utf8'),
    readFile(new URL('../scripts/build-teacher-coverage-selection.mjs', import.meta.url), 'utf8'),
    readFile(new URL('../../training/teacher_coverage.json', import.meta.url), 'utf8'),
  ]);
  const snapshot = pipeline.slice(pipeline.indexOf('snapshot() {'), pipeline.indexOf('\nfingerprint()'));
  assert.match(snapshot, /generate-synthetic-ir\.mjs/);
  assert.match(snapshot, /build-teacher-coverage-selection\.mjs/);
  assert.match(snapshot, /freeze-studio-teacher-cases\.mjs/);
  assert.match(pipeline, /while true/);
  assert.match(pipeline, /snapshot\n  after=/);
  assert.doesNotMatch(pipeline, /data\/external_pilot/);
  assert.match(selector, /data\/teacher\/native-synthetic\.ir\.jsonl/);
  assert.doesNotMatch(configText, /application:[^\"]+\.py/);
  for (const family of ['algo_prefix_sums', 'algo_window_sums', 'algo_top_k', 'algo_stable_unique',
    'algo_weighted_checksum', 'algo_adjacent_changes', 'algo_row_sums', 'algo_longest_true_run',
    'algo_merge_intervals', 'algo_staged_ranking', 'algo_algorithm_pipeline'])
    assert.match(configText, new RegExp(`family:${family}`));
});
