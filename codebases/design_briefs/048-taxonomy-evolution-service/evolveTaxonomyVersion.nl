---
args: { taxonomy: TaxonomyEvolutionRequest }
returns: TaxonomyEvolutionProposal
---
1. Bind fit = testMembershipAgainstVersionedDefinitions(taxonomy). Compare each record to current definitions and positive and negative examples; preserve overlapping or uncertain membership.
2. Bind defect = explainRepeatedBoundaryFailures(fit, taxonomy). Distinguish training guidance gaps from actual conceptual change. A rare concept may justify a category when evidence shows decision importance even if frequency is low.
3. Bind revision = draftStableIdentityRevision(defect, taxonomy). Preserve IDs for semantic renames; assign new identities where meaning changes. Check parent-child graph for cycles and contradictory boundaries, including concurrent steward edits.
4. Bind impact = replayOnlyAffectedRecords(revision, fit, taxonomy). Compute the exact changed-record set under baseline and candidate definitions. If a split cannot be reconstructed historically, report a coarser bridge rather than fabricated precision.
5. Bind compatibility = checkConsumerLabelContract(revision, taxonomy). Expose a single-label consumer's incompatibility with a new multi-label rule.
6. Bind release = authorizeTaxonomyPromotion(revision, impact, compatibility, taxonomy). Require steward approval, migration scope, dependent consumer compatibility, and versioned cache invalidation. Preserve old definitions for historical queries.
7. Return { taxonomyId: taxonomy.taxonomyId, fit, defect, revision, impact, compatibility, release, commitAllowed: false }.
