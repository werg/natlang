import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { scoreTurn, selectSamples } from '../scripts/eval-playground-turns.mjs';

const sample = (id, skill = 'read') => ({ id, family: 'lambda_scenario', skill,
  messages: [{ role: 'user', content: 'Do the task.' }], tools: [],
  target: { tool_calls: [{ function: { name: 'write', arguments: JSON.stringify({ path: 'return', value: 2 }) } }] } });

test('Node evaluator selects reproducible samples and verifies frozen content', async () => {
  const rows = [sample('a'), sample('b'), sample('c', 'reply')];
  const first = await selectSamples(rows, null, 1, 13);
  const second = await selectSamples(rows, null, 1, 13);
  assert.deepEqual(first.samples.map(row => row.id), second.samples.map(row => row.id));
  assert.equal(first.samples.length, 2);
  const directory = await mkdtemp(join(tmpdir(), 'natlang-eval-test-'));
  try {
    const path = join(directory, 'manifest.json');
    await writeFile(path, JSON.stringify(first.manifest));
    await assert.rejects(selectSamples(rows.map(row => row.id === first.samples[0].id ?
      { ...row, skill: 'changed' } : row), path), /Evaluation sample changed/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('Node evaluator scores tool, function calls, arguments, and terminal replies', () => {
  const action = sample('action');
  const calls = [{ function: { name: 'write', arguments: { path: 'return', value: 2 } } }];
  const scored = scoreTurn(action, calls, '');
  assert.equal(scored.exact, true);
  assert.equal(scored.right_tool, true);
  assert.equal(scored.reply, false);
  assert.equal(scoreTurn({ ...action, target: { tool_calls: [] } }, []).reply, true);
});
