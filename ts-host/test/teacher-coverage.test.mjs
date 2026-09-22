import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
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
