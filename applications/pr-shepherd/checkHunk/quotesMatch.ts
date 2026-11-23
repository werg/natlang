import type { Hunk, HunkReview } from '../types.js';

/** Exact postcondition verifier (auto-protected by ensures:): every finding must quote verbatim text from hunk.patch. */
export default function quotesMatch(review: HunkReview, hunk: Hunk): boolean {
  return review.findings.every(finding => finding.quote.trim().length > 0 && hunk.patch.includes(finding.quote));
}
