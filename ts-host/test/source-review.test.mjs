import test from 'node:test';
import assert from 'node:assert/strict';
import { SOURCE_REVIEWS, pendingSourceReview, sourceReviewReason } from '../dist/teacher/source-review.js';
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

test('criterion preservation guidance does not suggest new delegation at the depth limit', () => {
  assert.match(TOOLS_PROMPT, /preserve the parent's criterion and relevant context/);
  assert.match(TOOLS_PROMPT, /a question can express the same intent as a statement/);
  assert.doesNotMatch(TOOLS_PROMPT_AT_NL_DEPTH_LIMIT, /When delegating/);
});
