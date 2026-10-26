---
args: { left: WorkPackage, right: WorkPackage, costReviews: WorkPackageCostReview[], evidence: EvidencePage[] }
returns: { sharedDependencyIds: string[]; sourceRefs: string[]; sameOwnerScope: boolean; distinctWorkflowEvidence: string[]; estimateBasisRefs: string[] }
---
Require source evidence that names both package scopes or a common dependency identity. Separate common ownership from common work: a shared team alone is insufficient. Return exact dependency IDs, evidence references, whether owner scope is shared, any records showing different workflows, and cost-review estimate references applicable to the same work. Missing linkage remains empty with the evidence gaps explicit to the caller.
