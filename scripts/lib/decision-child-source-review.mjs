/**
 * Review a selected child result without letting its parent's aggregate result
 * erase an independently correct sibling. The child result remains the
 * admission-relevant value; parent output is retained as context only.
 */
export function reviewDecisionChildOutput({ expected, parentObserved, childObserved, review }) {
  if (expected === undefined) {
    return {
      accepted: false,
      parent_output_review: null,
      child_output_review: { accepted: false, errors: ['source_reference_missing'], entries: [] },
    };
  }
  const parentOutputReview = review(expected, parentObserved);
  const childOutputReview = review(expected, childObserved);
  return {
    accepted: childOutputReview?.accepted === true,
    parent_output_review: parentOutputReview,
    child_output_review: childOutputReview,
  };
}
