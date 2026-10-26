---
args: { left: WorkPackage, right: WorkPackage, costReviews: WorkPackageCostReview[], evidence: EvidencePage[] }
returns: SharedDependencyEvidence
---
Bind proof = qualifyDependencyEvidence(left, right, costReviews, evidence). Compare actual prerequisite identity, consuming mechanisms and user outcome. Keep distinct outcomes visible even when the same team or service is named. Return whether this is one operational dependency, both intervention IDs, exact evidence references, distinct-outcome risks, whether duplicated cost is counted once, and the blocker when evidence cannot establish the relationship.
