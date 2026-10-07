import assert from 'node:assert/strict';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { textworldQuest } from '../scripts/inline-curriculum/textworld.mjs';
import { missingDatasets } from './support/datasets.mjs';

const collectorUrl = process.env.NATLANG_COLLECTOR_JS ?
  pathToFileURL(process.env.NATLANG_COLLECTOR_JS).href :
  new URL('../dist/teacher/collector.js', import.meta.url).href;
const { executeProgram } = await import(collectorUrl);

const skip = missingDatasets('textworld');
const [playable] = skip ? [] : textworldQuest(17, 0);
const expected = playable?.semantics.expected;
const commands = playable?.curriculum.evidence.background[1]
  .replace(/^winning commands: /, '').split(' | ');
const executeOptions = { rootSeed: 17, systemPrompt: 'Use tools to complete the task.', contextTokens: 16384, maxTurns: 4 };

test('TextWorld rejects returning a literal certificate before the host world completes', { skip }, async () => {
  const run = await executeProgram(playable, async () => ({
    calls: [['return_result', { status: 'success', value: expected }]],
  }), executeOptions);

  assert.equal(run.outcome.oracle.accepted, true, 'the string itself matches the answer oracle');
  assert.equal(run.outcome.host_completion_certificate, null);
  assert.equal(run.outcome.checks.world, false);
  assert.equal(run.outcome.accepted, false);
});

test('TextWorld accepts the certificate after trusted world actions actually complete the quest', { skip }, async () => {
  let turn = 0;
  const run = await executeProgram(playable, async () => turn++ === 0 ? ({
    calls: [['eval', { code: `for (const command of ${JSON.stringify(commands)}) world.act(command); world.certificate()` }]],
  }) : ({ calls: [['return_result', { status: 'success', value: expected }]] }), executeOptions);

  assert.equal(run.outcome.host_completion_certificate, expected);
  assert.equal(run.outcome.checks.world, true);
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome.checks));
});
