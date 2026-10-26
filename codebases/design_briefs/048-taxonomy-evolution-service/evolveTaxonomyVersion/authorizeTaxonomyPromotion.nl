---
args: { revision: TaxonomyDefinitionRevision, impact: TaxonomyRecordImpact, compatibility: TaxonomyConsumerCompatibility[], taxonomy: TaxonomyEvolutionRequest }
returns: TaxonomyPromotionDecision
---
Bind stewardApproval to the exact definition version, affected-record set, and historical bridge. Block promotion while branch-boundary conflicts, consumer incompatibility, or unapproved migration scope remains. Preserve prior definitions and invalidate only caches whose record classification actually changes.
