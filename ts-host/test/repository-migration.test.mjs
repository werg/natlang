import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { RepositoryMigration } from '../../applications/repository_migration.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const path = fileURLToPath(new URL('../../codebases/repository_migration/migrate.nl', import.meta.url));
const source = {
  'lib.mjs': 'export function sum(a, b) { return a + b; }\n',
  'caller.mjs': "import { sum } from './lib.mjs';\nexport const total = sum(2, 3);\n",
  'test.mjs': "import { strict as assert } from 'node:assert';\nimport { total } from './caller.mjs';\nassert.equal(total, 5);\n",
};

test('natlang plans a checked migration without editing the original repository', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  for (const [name, text] of Object.entries(source)) await writeFile(join(root, name), text);
  const repository = new RepositoryMigration(root, { files: Object.keys(source),
    checks: [{ id: 'scenario', argv: [process.execPath, '--test', 'test.mjs'] }] });
  await repository.open();
  const host = new NatlangHost({ host: { repository,
    drainEvents: () => repository.drainEvents() } });
  try {
    const result = await host.run({ source: { kind: 'file', path },
      inputs: { request: 'Rename sum to add while preserving the calculation', query: 'sum' },
      modelTurn: request => {
        const turn = request;
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function migrate(')) return evalTurn(turn,
          'const snapshot = await inspect();\n' +
          'const uses = await search(query, snapshot.revision);\n' +
          'const patch = await propose(request, snapshot, uses);\n' +
          'const candidate = await apply(snapshot.revision, patch);\n' +
          'const checks = await validate(candidate.revision);\n' +
          'await report(candidate.revision, checks)');
        return evalTurn(turn, JSON.stringify([
          { path: 'lib.mjs', old: 'function sum(', new: 'function add(' },
          { path: 'caller.mjs', old: '{ sum }', new: '{ add }' },
          { path: 'caller.mjs', old: 'sum(2, 3)', new: 'add(2, 3)' },
        ]));
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'reviewable');
    assert.equal(result.value.changed.length, 2);
    const base = repository.snapshot().revision;
    assert.equal(await readFile(join(root, 'caller.mjs'), 'utf8'), source['caller.mjs']);
    assert.throws(() => repository.apply(base, [{ path: 'caller.mjs', old: 'sum', new: 'add' }]),
      /ambiguous/);
    assert.equal(await readFile(join(root, 'lib.mjs'), 'utf8'), source['lib.mjs']);
  } finally { host.close(); await rm(root, { recursive: true, force: true }); }
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
