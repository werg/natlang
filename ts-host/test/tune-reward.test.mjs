import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { scorable, scoreRoot, verifierDigest } from '../dist/tune/reward.js';

test('a root scores by its exact oracle without a program verifier', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-reward-'));
  assert.equal(verifierDigest(root), null);
  assert.deepEqual(await scoreRoot(root, { expected: ['a'] }, ['a'], false), { quality: 1, source: 'root-oracle' });
  assert.deepEqual(await scoreRoot(root, { expected: ['a'] }, ['a'], true), { quality: 0, source: 'root-oracle' });
  assert.equal(await scoreRoot(root, {}, 'x', false), null);
  assert.equal(await scorable(root, {}), false);
});

test('a program verifier grades roots in [0, 1] from their reference data and binds its digest', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-reward-'));
  writeFileSync(join(root, 'tune.reward.mjs'),
    'export default (value, example) => example.reference.gains[value] / Math.max(...Object.values(example.reference.gains));\n');
  const example = { reference: { gains: { keep: 0, refine: 1, propose: 4 } } };
  const scored = await scoreRoot(root, example, 'refine', false);
  assert.equal(scored.quality, .25);
  assert.equal(scored.source, 'root-verifier');
  assert.equal(scored.verifier, verifierDigest(root));
  assert.equal((await scoreRoot(root, example, 'refine', true)).quality, 0);
  assert.equal(await scorable(root, {}), true);
  writeFileSync(join(root, 'tune.reward.mjs'), 'export default () => 2;\n');
  await assert.rejects(scoreRoot(root, example, 'refine', false), /number in \[0, 1\]/);
});

test('the learned optimizer grades its own stages by their share of the measured improvement', async () => {
  const root = new URL('../../applications/learned-optimizer/', import.meta.url).pathname;
  const operator = { function: 'adapt/chooseOperator.nl', expected: 'local-refinement',
    reference: { incumbentLoss: 2, losses: { 'learned-proposal': 1.5, 'local-refinement': 1, retrieval: 2.5 } } };
  assert.equal((await scoreRoot(root, operator, 'local-refinement', false)).quality, 1);
  assert.equal((await scoreRoot(root, operator, 'learned-proposal', false)).quality, .5);
  assert.equal((await scoreRoot(root, operator, 'retrieval', false)).quality, 0);
  assert.equal((await scoreRoot(root, operator, 'keep', false)).quality, 0);
  const stop = { function: 'adapt/shouldStop.nl', expected: false, reference: { stop: false, remainingShare: .25 } };
  assert.equal((await scoreRoot(root, stop, true, false)).quality, .75);
  const focus = { function: 'adapt/chooseFocus.nl', expected: ['a', 'b'], reference: { focus: ['a', 'b'] } };
  assert.equal((await scoreRoot(root, focus, ['a'], false)).quality, 2 / 3);
});
