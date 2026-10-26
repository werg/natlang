---
args: { documentation: DocumentationEditionRequest }
returns: DocumentationEditionProposal
---
1. Bind promises = extractVersionedPromises(documentation). Each claim carries exact page span, audience, normative status, and applicable release.
2. Bind probes = designReproducibleBehaviorProbes(promises, documentation). Use supplied observations where available; otherwise prepare bounded probes with environment and release identity rather than pretending they ran.
3. Bind mismatches = adjudicateClaimAgainstAuthority(promises, probes, documentation). Keep product defects separate from stale wording, intentional changes, and unresolved intent.
4. Bind patch = draftCoordinatedDocumentationRepair(mismatches, documentation). Preserve unaffected claims and exact patch bases; deprecated services require a reviewed replacement, and secret-bearing examples are blocked from export.
5. Bind rebase = revalidatePageEditsAgainstReviewBase(patch, documentation). Recheck changed page spans after concurrent edits; separate translation edition lag from translation quality.
6. Bind release = verifyOneVersionDocumentationBundle(promises, probes, mismatches, rebase, documentation). Re-extract changed promises and revalidate linked examples at the same target release. Return { reviewId: documentation.reviewId, promises, probes, mismatches, patch, rebase, release, commitAllowed: false }.
