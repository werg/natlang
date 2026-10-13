import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { CallStore } from '../dist/calls/store.js';
import { interpreter, lambda, nl } from './support/natlang.mjs';

// Runs of one kind share a logical name (`curriculum-reference`, the default `native-run`); each run is its own root
// call, so the store keeps every run's root record and the children that link to it.
const agent = async session => {
  const child = String(session.lam.body ?? '').includes('Inspect');
  await session.applyAsync('eval', { code: child ? 'return value * 2' : 'return await inspect(value)' });
};
const root = value => lambda({ type: '(value: number) => number', instructions: 'Ask the helper.', args: { value },
  codebase: { inspect: nl('inspect', { args: { value: 'number' }, returns: 'number', instructions: 'Inspect the value.' }) } });

for (const [name, options] of [['a shared label', { runLabel: 'curriculum-reference' }], ['the default run', {}]]) {
  test(`two runs under ${name} keep two root calls with their own children`, async () => {
    const dir = mkdtempSync(join(tmpdir(), 'natlang-run-identity-'));
    const store = CallStore.open(dir);
    try {
      const runtimes = [];
      for (const value of [3, 5]) {
        const runtime = interpreter({ calls: store, agent, ...options });
        const result = await runtime.run(root(value));
        assert.equal(result.outcome.kind, 'done');
        assert.equal(result.value, value * 2);
        runtimes.push(runtime);
      }
      const label = options.runLabel ?? 'native-run';
      const [first, second] = runtimes.map(runtime => runtime.options.runId);
      assert.notEqual(first, second, 'each run has its own root call ID');
      for (const [index, runId] of [first, second].entries()) {
        assert.equal(runtimes[index].options.runLabel, label);
        assert.equal(runtimes[index].trace.events.find(event => event.kind === 'invocation')?.call_id, runId);
        const record = store.call(runId);
        assert.ok(record, `root ${runId} is in the store`);
        assert.equal(record.run_label, label);
        assert.equal(store.value(record.inputs.value), [3, 5][index]);
        const row = store.db.prepare('SELECT run_label FROM calls WHERE call_id = ?').get(runId);
        assert.equal(row.run_label, label);
        const children = store.calls().filter(call => call.parent_call_id === runId);
        assert.equal(children.length, 1, `root ${runId} keeps its child`);
        assert.equal(children[0].definition_name, 'inspect');
        assert.equal(store.value(store.call(children[0].call_id).output), [6, 10][index]);
      }
      const labelled = store.db.prepare('SELECT call_id FROM calls WHERE run_label = ?').all(label).map(row => row.call_id).sort();
      assert.deepEqual(labelled, [first, second].sort());
    } finally { store.close(); rmSync(dir, { recursive: true, force: true }); }
  });
}

test('an explicit run ID is kept, and seeds follow the label rather than the unique ID', async () => {
  const explicit = interpreter({ calls: false, runId: 'teacher-run-7' });
  assert.equal(explicit.options.runId, 'teacher-run-7');
  assert.equal(explicit.options.runLabel, undefined);
  const { runSeedKey } = await import('../dist/native/runtime.js');
  assert.equal(runSeedKey(explicit.options), 'teacher-run-7');
  assert.equal(runSeedKey(interpreter({ calls: false }).options), 'native-run');
  assert.equal(runSeedKey(interpreter({ calls: false, runId: 'curriculum-reference:replay:x:1', runLabel: 'curriculum-reference' }).options),
    'curriculum-reference');
  assert.equal(runSeedKey(interpreter({ calls: false, runLabel: 'l', seedId: 's' }).options), 's');
});

test('a store from before run labels gains the column and keeps its rows', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-run-identity-migrate-'));
  try {
    let store = CallStore.open(dir);
    store.db.exec('DROP INDEX calls_run_label');
    store.db.exec('ALTER TABLE calls DROP COLUMN run_label');
    store.db.prepare(`INSERT INTO calls (call_id, definition_id, definition_name, definition_key, executor, outcome, record_hash)
      VALUES ('old-run', 'native:root', 'root', 'k', 'agent', 'done', '')`).run();
    store.close();
    store = CallStore.open(dir);
    try {
      const columns = store.db.prepare('PRAGMA table_info(calls)').all().map(column => column.name);
      assert.ok(columns.includes('run_label'));
      assert.equal(store.db.prepare('SELECT run_label FROM calls WHERE call_id = ?').get('old-run').run_label, null);
      const runtime = interpreter({ calls: store, agent, runLabel: 'after-migration' });
      await runtime.run(root(1));
      assert.equal(store.call(runtime.options.runId).run_label, 'after-migration');
    } finally { store.close(); }
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
