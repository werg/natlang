import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('a named function runs on the model its frontmatter names, and on the default model when none has that name', async () => {
  const judge = defineNatlang('---\nargs:\n  text: string\nreturns: string\nmodel: small\n---\nJudge text.\n', { name: 'judge' });
  const write = defineNatlang('---\nargs:\n  text: string\nreturns: string\n---\nWrite about text.\n', { name: 'write' });
  const big = scriptedModel(() => 'return "big";'), small = scriptedModel(() => 'return "small";');
  const runtime = createNatlangRuntime({ model: big.driver, models: { small: small.driver } });
  assert.equal(await runtime.run(() => judge('x')), 'small');
  assert.equal(await runtime.run(() => write('x')), 'big');
  assert.equal(await createNatlangRuntime({ model: big.driver }).run(() => judge('x')), 'big');
  assert.throws(() => defineNatlang('---\nreturns: string\nmodel: two words\n---\nx\n'), /model must name one of the runtime's models/);
});
