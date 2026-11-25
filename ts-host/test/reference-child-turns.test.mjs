// The reference driver counts a child's scripted turns per call (its invocation id), so two calls with the same
// opening each play the entry's turns from the first (authored trajectories: the same judgment asked twice).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { replayReference } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

test('two calls with one opening each play a one-turn scripted entry', async () => {
  const files = {
    'root.nl': '---\nargs:\n  note: string\nreturns: string[]\n---\nAsk echo about note twice.\n',
    'root/echo.nl': '---\nargs:\n  note: string\nreturns: string\n---\nRepeat note in your own words.\n',
  };
  const record = { version: 'natlang.program/2', id: 'twice', kind: 'lambda_source', source: 'fixture', split: 'train',
    source_ids: ['twice'], source_groups: ['twice'], license: 'test', family: 'fixture',
    semantics: { root: 'root.nl', files, inputs: { note: 'the gate is locked' }, expected: ['again', 'again'], oracle: 'exact' },
    curriculum: { reference: {
      root: [['eval', { code: 'const first = await echo(note);\nconst second = await echo(note);\nreturn [first, second];', finish: true }]],
      children: [{ match: ['You are inside this call: echo('], calls: [['return_result', { status: 'success', value: 'again' }]] }] } } };
  const { run } = await replayReference(record, TOOLS_PROMPT);
  assert.equal(run.outcome.status, 'done', JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 600));
  assert.deepEqual(run.outcome.value, ['again', 'again']);
});
