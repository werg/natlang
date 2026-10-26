---
args: { request: DocumentationEditionRequest, proposal: DocumentationEditionProposal, targets: DocumentationPagePatchTarget[], receipt: DocumentationPublicationReceipt }
returns: DocumentationPublicationAssessment
---
1. Require receipt.reviewId and targetRelease to match the reviewed edition. Publication is blocked if release status is not ready, unresolved mismatches or broken examples remain, stale spans or translation-edition lag remain, or blockedSecretSpans is nonempty.
2. Bind every page receipt to its exact page target and currentBaseVersion. The appliedPatch must be the exact approved page patch and source spans must remain those reviewed. A concurrent base change requires rebase and claim review before the edit can be treated as current.
3. A page with status partial, failed, or unknown is not published as current. Preserve its observed state and return its page ID for repair/reconciliation. Do not claim a whole release from some successful page writes.
4. Return canRecordAsCurrent only when the entire reviewed bundle is ready, secret scan is clear, all page edits match the approved patch and current base, and the actual publication receipt proves each resulting edition.
