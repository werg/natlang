import assert from 'node:assert/strict';
import { reviewDecisionChildOutput } from './lib/decision-child-source-review.mjs';

const exactReview = (expected, observed) => ({
  accepted: expected === observed,
  errors: expected === observed ? [] : ['source_label_mismatch'],
  entries: [],
});

const correctSibling = reviewDecisionChildOutput({ expected: 'time', parentObserved: 'distance',
  childObserved: 'time', review: exactReview });
assert.equal(correctSibling.accepted, true,
  'the exact child is eligible even when its parent aggregate chose a different label');
assert.equal(correctSibling.parent_output_review.accepted, false,
  'the parent disagreement is preserved as context');
assert.equal(correctSibling.child_output_review.accepted, true);

const wrongSibling = reviewDecisionChildOutput({ expected: 'time', parentObserved: 'distance',
  childObserved: 'timezone', review: exactReview });
assert.equal(wrongSibling.accepted, false,
  'a wrong child remains rejected even when the parent also disagrees');
assert.equal(wrongSibling.child_output_review.accepted, false);

const missingSource = reviewDecisionChildOutput({ expected: undefined, parentObserved: 'time',
  childObserved: 'time', review: exactReview });
assert.equal(missingSource.accepted, false,
  'a matching output cannot replace a missing source reference');
assert.equal(missingSource.child_output_review.accepted, false);

console.log('decision child source review regression checks passed');
