import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { RepositoryMigration, migrate } from '../../applications/dist/migration/index.js';
import { scriptedModel } from './support/natlang.mjs';

const source = {
  'lib.mjs': 'export function sum(a, b) { return a + b; }\n',
  'caller.mjs': "import { sum } from './lib.mjs';\nexport const total = sum(2, 3);\n",
  'test.mjs': "import { strict as assert } from 'node:assert';\nimport { total } from './caller.mjs';\nassert.equal(total, 5);\n",
};

test('natlang plans, checks and repairs a migration without editing the original repository', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  for (const [name, text] of Object.entries(source)) await writeFile(join(root, name), text);
  const repository = new RepositoryMigration(root, { files: Object.keys(source),
    checks: [{ id: 'scenario', argv: [process.execPath, '--test', 'test.mjs'] }] });
  await repository.open();
  let proposals = 0;
  const driver = scriptedModel(opening => {
    proposals++;
    // The first candidate misses the callers; the repair reads the failed checks and fixes them.
    return opening.includes('failed: Validation = missing') ?
      'result = [{ path: "lib.mjs", old: "function sum(", new: "function add(" }]' :
      'result = [{ path: "caller.mjs", old: "{ sum }", new: "{ add }" }, { path: "caller.mjs", old: "sum(2, 3)", new: "add(2, 3)" }]';
  });
  try {
    const result = await createNatlangRuntime({ model: driver.driver }).run(() =>
      migrate(repository, 'Rename sum to add while preserving the calculation', 'sum'));
    assert.equal(result.status, 'reviewable', JSON.stringify(result.checks));
    assert.equal(proposals, 2);
    assert.equal(result.changed.length, 2);
    const base = repository.snapshot().revision;
    assert.equal(await readFile(join(root, 'caller.mjs'), 'utf8'), source['caller.mjs']);
    assert.throws(() => repository.apply(base, [{ path: 'caller.mjs', old: 'sum', new: 'add' }]),
      /ambiguous/);
    assert.equal(await readFile(join(root, 'lib.mjs'), 'utf8'), source['lib.mjs']);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('failed checks and stale patch context remain visible for repair', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  for (const [name, text] of Object.entries(source)) await writeFile(join(root, name), text);
  const repository = new RepositoryMigration(root, { files: Object.keys(source),
    checks: [{ id: 'scenario', argv: [process.execPath, '--test', 'test.mjs'] }] });
  try {
    const base = (await repository.open()).revision;
    const broken = repository.apply(base, [
      { path: 'lib.mjs', old: 'function sum(', new: 'function add(' },
    ]);
    const validation = await repository.validate(broken.revision);
    assert.equal(validation.status, 'failed', JSON.stringify(validation));
    assert.equal(repository.report(broken.revision, validation).status, 'checks-failed');
    const repaired = repository.apply(broken.revision, [
      { path: 'caller.mjs', old: '{ sum }', new: '{ add }' },
      { path: 'caller.mjs', old: 'sum(2, 3)', new: 'add(2, 3)' },
    ]);
    assert.equal((await repository.validate(repaired.revision)).status, 'passed');
    assert.throws(() => repository.apply(broken.revision, [
      { path: 'lib.mjs', old: 'function sum(', new: 'function add(' },
    ]), /missing/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('long checks stream output and retain a bounded diagnostic tail', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  await writeFile(join(root, 'lib.mjs'), source['lib.mjs']);
  const repository = new RepositoryMigration(root, { files: ['lib.mjs'], checks: [
    { id: 'verbose', argv: [process.execPath, '-e',
      "process.stdout.write('x'.repeat(2_000_000))"] },
  ] });
  try {
    const snapshot = await repository.open();
    const checked = await repository.validate(snapshot.revision);
    assert.equal(checked.status, 'passed');
    assert.equal(checked.checks[0].output_bytes, 2_000_000);
    assert.equal(checked.checks[0].truncated, true);
    assert.equal(checked.checks[0].output.length, 4000);
  } finally { await rm(root, { recursive: true, force: true }); }
});
