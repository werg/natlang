import type { Hunk, HunkReview } from '../types.js';

/** Exact postcondition helper (auto-protected from edits by `ensures:`): every finding must quote text present in the hunk. */
export default function quotesMatch(review: HunkReview, hunk: Hunk): boolean {
  return review.findings.every(finding => finding.quote.trim().length > 0 && hunk.patch.includes(finding.quote));
}
