import test from 'node:test';
import assert from 'node:assert/strict';
import { SOURCE_REVIEWS, anliReviewText, pendingSourceReview, sourceReviewReason } from '../dist/teacher/source-review.js';
import { quarantineReason } from '../dist/teacher/curriculum-policy.js';
import { TOOLS_PROMPT, TOOLS_PROMPT_AT_NL_DEPTH_LIMIT } from '../dist/native/prompt.js';

test('pending source reviews match current and legacy identities within their dataset', () => {
  for (const review of SOURCE_REVIEWS) for (const id of [review.id, ...review.aliases]) {
    assert.equal(pendingSourceReview(review.dataset, id), review);
    assert.equal(sourceReviewReason({ dataset: review.dataset, dataset_records: [id] }), 'source_review_pending');
    assert.equal(sourceReviewReason({ dataset: review.dataset, source_ids: [id] }), 'source_review_pending');
    assert.equal(sourceReviewReason({ dataset: 'sms_spam', dataset_records: [id] }), undefined);
    assert.equal(quarantineReason({ dataset: review.dataset, dataset_records: [id], semantics: {} }), 'source_review_pending');
  }
  assert.equal(sourceReviewReason({ dataset: 'banking77', dataset_records: ['unaffected'] }), undefined);
  assert.equal(sourceReviewReason({ dataset_records: [SOURCE_REVIEWS[0].id] }), undefined);
});

test('legacy αNLI batches without source IDs are held by complete visible identity', () => {
  const review = SOURCE_REVIEWS.find(item => item.dataset === 'anli');
  const [beginning, ending, a, b] = review.text.split('\n');
  const story = { id: 'N1', beginning, ending, a, b };
  const record = { curriculum: { family: 'anli_batch' }, semantics: { inputs: { stories: [story] } } };
  assert.equal(anliReviewText(story), review.text);
  assert.equal(sourceReviewReason(record), 'source_review_pending');
  assert.equal(quarantineReason(record), 'source_review_pending');
  record.semantics.inputs.stories[0].b = 'A different, supported hypothesis.';
  assert.equal(sourceReviewReason(record), undefined);
});

test('FOLIO story holds cover single conclusions and legacy batches by source group or shape', () => {
  for (const family of ['folio_entailment', 'folio_batch']) {
    assert.equal(sourceReviewReason({ curriculum: { family }, source_groups: ['folio:story:337'] }), 'source_review_pending');
    assert.equal(sourceReviewReason({ curriculum: { family, shape: 'story162' } }), 'source_review_pending');
    assert.equal(sourceReviewReason({ curriculum: { family, shape: 'story56' } }), 'source_review_pending');
    assert.equal(sourceReviewReason({ curriculum: { family }, source_groups: ['folio:story:8'] }), 'source_review_pending');
    assert.equal(sourceReviewReason({ curriculum: { family, shape: 'story348' } }), 'source_review_pending');
    assert.equal(sourceReviewReason({ curriculum: { family }, source_groups: ['folio:story:377'] }), 'source_review_pending');
    assert.equal(sourceReviewReason({ curriculum: { family, shape: 'story999' } }), undefined);
  }
});

test('criterion preservation guidance does not suggest new delegation at the depth limit', () => {
  assert.match(TOOLS_PROMPT, /preserve the parent's criterion and relevant context/);
  assert.match(TOOLS_PROMPT, /a question can express the same intent as a statement/);
  assert.doesNotMatch(TOOLS_PROMPT_AT_NL_DEPTH_LIMIT, /When delegating/);
});

test('movie schema version 3 is held until migrated to world-derived ownership', () => {
  const record = { curriculum: { family: 'commaqa_question', family_version: 3 }, semantics: {} };
  assert.equal(quarantineReason(record), 'unverified_movie_schema_contract');
  record.curriculum.family_version = 4;
  assert.equal(quarantineReason(record), undefined);
});
