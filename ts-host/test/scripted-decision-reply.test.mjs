// Scripted drivers (reference replay, authoring oracle) answer a decision readout's tool-free generation request with
// the value's JSON text, the surface agent.ts generatedDecision reads since 61c6c3614; a return_result call there is
// rejected as decision-generation-invalid. The fixture is an authored folder task whose reference eval calls decide.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { replayReference } from '../dist/teacher/curriculum.js';
import { scriptedSuccess } from '../dist/native/decision.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const record = JSON.parse(readFileSync(new URL('./fixtures/authored-decide-reference.json', import.meta.url), 'utf8'));

test('scriptedSuccess answers a tool-free request with JSON text and a tool-loop request with return_result', () => {
  assert.deepEqual(scriptedSuccess({ tools: [] }, 'keep'), { text: '"keep"', calls: [] });
  assert.deepEqual(scriptedSuccess({ tools: [{ type: 'function' }] }, { a: 1 }),
    { calls: [['return_result', { status: 'success', value: { a: 1 } }]] });
});

test('a reference that calls decide on a readout child replays to the expected folder and result', async () => {
  const { run } = await replayReference(record, TOOLS_PROMPT);
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 600));
  assert.deepEqual(run.outcome.value, { keep: 1, toss: 1 });
});

test('a decision-readout root replays its scripted return_result value as the decision text', async () => {
  const root = '---\nargs:\n  note: string\nreturns: \'"keep" | "toss"\'\nreadout: decision\n---\nDecide whether note is worth keeping.\n';
  const stage = { version: 'natlang.program/2', id: 'stage-root', kind: 'lambda_source', source: 'fixture', split: 'train',
    source_ids: ['stage-root'], source_groups: ['stage-root'], license: 'test', family: 'fixture',
    semantics: { root: 'judge/keep.nl', files: { 'judge.nl': '---\nargs: {}\nreturns: string\n---\nReturn "x".\n', 'judge/keep.nl': root },
      inputs: { note: 'a receipt for the tax return' }, expected: 'keep', oracle: 'exact' },
    curriculum: { reference: { root: [['return_result', { status: 'success', value: 'keep' }]] } } };
  const { run } = await replayReference(stage, TOOLS_PROMPT);
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 600));
  assert.equal(run.outcome.value, 'keep');
});
