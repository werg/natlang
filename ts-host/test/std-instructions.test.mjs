// @natlang/std instructionsOf / stepsOf (crisp) and the shipped stepKind.nl judgment, shared by the idiom reviewer and
// the program migrator.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { instructionsOf, stepKind, stepsOf } from '../dist/std/index.js';

test('instructionsOf: front matter, declaration without readout, numbered steps (CRLF too); paragraphs otherwise', () => {
  const read = instructionsOf('---\r\nargs:\r\n  x: string\r\nreturns: V\r\nreadout: decision\r\n---\r\nIntro.\r\n1. One.\r\n   more\r\n2. Two:\r\n```ts\r\n3. not a step\r\n```\r\n');
  assert.equal(read.declaration, 'args:\n  x: string\nreturns: V');
  assert.deepEqual(read.steps, ['Intro.', '1. One.\n   more', '2. Two:\n```ts\n3. not a step\n```']);
  assert.deepEqual(stepsOf('First\nline.\n\nSecond.'), ['First\nline.', 'Second.']);
  assert.throws(() => instructionsOf('no front matter'), /front matter/);
});

test('stepKind is a shipped decision over the four step kinds', async () => {
  const model = async () => ({ text: '"judgment"', calls: [] });
  let options;
  model.decide = async req => { options = req.options; return { log_probs: req.options.map(o => Math.log(o === '"mechanism"' ? 0.9 : 0.03)) }; };
  const kind = await createNatlangRuntime({ model }).run(() => stepKind('Sort the rows by id.', 'args:\n  rows: Row[]\nreturns: string', 'report.nl'));
  assert.equal(kind, 'mechanism');
  assert.deepEqual([...options].sort(), ['"guard-for-type"', '"judgment"', '"mechanism"', '"redundant"']);
});
