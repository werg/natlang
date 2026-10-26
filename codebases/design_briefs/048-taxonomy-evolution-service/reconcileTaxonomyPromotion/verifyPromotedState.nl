---
args: { book: TaxonomyReleaseBook, proposal: TaxonomyEvolutionProposal, receipt: TaxonomyPromotionReceipt, expectedDefinitionDigest: string }
returns: TaxonomyActivationAssessment
---
1. Require taxonomyId, candidate version, active prior version, and expected definition digest to match the proposal and current release book. A version name alone does not bind the definition contents.
2. Require proposal.release.status "ready", no approvalsNeeded, historical definitions retained, and no cyclic or contradictory boundaries in the proposed definition revision. The proposal's review decision is not a publish receipt.
3. For each invalidated record ID, require an actual replayedRecordId in the receipt; unresolvedRecordIds remain open and block promotion when they intersect that affected set. Preserve ambiguous historical bridges as explicit uncertainty.
4. Require an actual acceptance receipt from every consumer that the proposal marks compatible. Return all missing replays, incompatible or unacknowledged consumers, digest/version mismatches and approval blockers.
