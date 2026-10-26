---
args: { intervention: InterventionMechanism, candidatePackage: WorkPackage, evidence: EvidencePage[] }
returns: WorkPackageCostReview
---
Bind basis = reviewWorkCostBasis(intervention, candidatePackage, evidence). Attach the returned exact estimate identities, supported range, confidence and blockers to this package. Attribute each dependency to the intervention that requires it and retain the source that establishes its owner. A package with no authorized estimate remains unresolved and cannot be treated as zero-cost. Return the package, basis review and dependency attributions together.
