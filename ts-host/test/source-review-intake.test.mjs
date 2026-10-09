import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { configureSourceReview, noteSourceItem, notedSourceItems, sourceReviewMode, writeIntakeReviews } from '../scripts/inline-curriculum/source-review-intake.mjs';
import { pendingSourceReview, SOURCE_REVIEWS } from '../dist/teacher/source-review.js';

const item = { dataset: 'sst2', id: 'intake-item-1', visible: 'The film is fine. Nothing else is said.', annotated_label: 'negative',
  contract: 'Label the sentiment of the sentence.', answer_format: 'negative | positive' };
const hold = { recommendation: 'hold', concern: 'label-disagrees-with-source', reason: 'The text calls the film fine, which supports a positive label.',
  evidence: [{ quote: 'The film is fine.' }], proposed_entry: { reason: 'The text calls the film fine, which supports a positive label.' }, confidence: 'high' };
const scratch = () => mkdtempSync(join(tmpdir(), 'natlang-intake-review-'));
const scripted = answer => { const calls = []; return { calls, executor: { executor: 'scripted', call: async name => { calls.push(name); return answer; } } }; };

test('the default is crisp: nothing is noted, no model is called and no file is written', async () => {
  configureSourceReview({}, {});
  assert.equal(sourceReviewMode(), 'crisp');
  noteSourceItem(item);
  assert.deepEqual(notedSourceItems(), []);
  const dir = scratch();
  try {
    const { calls, executor } = scripted(hold);
    assert.equal(await writeIntakeReviews(join(dir, 'review.jsonl'), { executor }), null);
    assert.equal(calls.length, 0);
    assert.equal(existsSync(join(dir, 'review.jsonl')), false);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('the mode comes from the flag over the environment and is validated', () => {
  assert.equal(configureSourceReview({ 'source-review-mode': 'shadow' }, { NATLANG_SOURCE_REVIEW_MODE: 'nl' }), 'shadow');
  assert.equal(configureSourceReview({}, { NATLANG_SOURCE_REVIEW_MODE: 'nl' }), 'nl');
  assert.throws(() => configureSourceReview({ 'source-review-mode': 'always' }, {}), /one of crisp, nl, shadow/);
  assert.throws(() => configureSourceReview({ 'source-review-mode': 'nl', 'source-review-limit': '0' }, {}), /positive/);
  configureSourceReview({}, {});
});

for (const mode of ['nl', 'shadow']) {
  test(`${mode} writes recommendations beside the output and changes no admission`, async () => {
    const dir = scratch();
    try {
      const before = SOURCE_REVIEWS.length, pendingBefore = pendingSourceReview(item.dataset, item.id);
      configureSourceReview({ 'source-review-mode': mode }, {});
      noteSourceItem(item); noteSourceItem(item);
      assert.equal(notedSourceItems().length, 1, 'an item is noted once');
      const { calls, executor } = scripted(hold);
      const path = join(dir, 'shard.jsonl.source-review.jsonl');
      const summary = await writeIntakeReviews(path, { executor });
      assert.deepEqual([summary.reviewed, summary.hold, summary.errors], [1, 1, 0]);
      assert.ok(calls.length >= 1 && calls.every(name => name === 'reviewSourceItem'));
      const [line] = readFileSync(path, 'utf8').trim().split('\n').map(text => JSON.parse(text));
      assert.equal(line.kind, 'item');
      assert.equal(line.recommendation.recommendation, 'hold');
      assert.equal(line.training_admission, false);
      assert.match(line.reviewer.reviewer_hash, /^natlang@[0-9a-f]{16}$/);
      assert.equal(SOURCE_REVIEWS.length, before);
      assert.equal(pendingSourceReview(item.dataset, item.id), pendingBefore, 'the registry does not move');
    } finally { configureSourceReview({}, {}); rmSync(dir, { recursive: true, force: true }); }
  });
}

test('an answer that breaks the typed contract is recorded as excluded; items with a standing entry are not asked again', async () => {
  const dir = scratch();
  try {
    configureSourceReview({ 'source-review-mode': 'nl' }, {});
    const standing = SOURCE_REVIEWS.find(review => review.status === 'pending');
    noteSourceItem(item);
    noteSourceItem({ ...item, dataset: standing.dataset, id: standing.id });
    const { calls, executor } = scripted({ ...hold, evidence: [{ quote: 'a sentence the text does not contain' }] });
    const path = join(dir, 'review.jsonl');
    const summary = await writeIntakeReviews(path, { executor });
    assert.deepEqual([summary.reviewed, summary.errors, summary.standing], [0, 1, 1]);
    assert.equal(calls.length, 1, 'only the new item was asked');
    const [line] = readFileSync(path, 'utf8').trim().split('\n').map(text => JSON.parse(text));
    assert.equal(line.excluded, 'transport_error');
    assert.match(line.error, /copied exactly from item\.visible/);
  } finally { configureSourceReview({}, {}); rmSync(dir, { recursive: true, force: true }); }
});

test('the limit bounds the noted items', () => {
  configureSourceReview({ 'source-review-mode': 'nl', 'source-review-limit': '2' }, {});
  for (let index = 0; index < 5; index++) noteSourceItem({ ...item, id: `limit-${index}` });
  assert.equal(notedSourceItems().length, 2);
  configureSourceReview({}, {});
});

test('the written lines are what scripts/source_review.py receipt reads', { skip: spawnSync('python3', ['--version']).status !== 0 && 'needs python3' }, async () => {
  const dir = scratch();
  try {
    configureSourceReview({ 'source-review-mode': 'nl' }, {});
    noteSourceItem(item);
    const path = join(dir, 'review.jsonl');
    await writeIntakeReviews(path, { executor: scripted(hold).executor });
    const script = new URL('../../scripts/source_review.py', import.meta.url).pathname;
    const result = spawnSync('python3', ['-I', script, '--receipts', join(dir, 'receipts'), 'receipt', path], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout.trim().split('\n')[0]).training_admission, false);
  } finally { configureSourceReview({}, {}); rmSync(dir, { recursive: true, force: true }); }
});
