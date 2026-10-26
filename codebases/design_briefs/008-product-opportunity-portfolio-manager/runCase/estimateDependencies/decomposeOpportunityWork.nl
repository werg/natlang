---
args: { intervention: InterventionMechanism, evidence: EvidencePage[] }
returns: WorkPackageDecomposition
---
Separate build, rollout, operations, and enabling work only when evidence and intervention description support distinct packages. For each candidate package call attributeWorkPackage(intervention, candidatePackage, evidence). Use the returned package, cost-basis review, and dependency attributions; leave unauthorized or missing estimates unfilled instead of copying neighboring costs. Return { packages: the distinct source-attributed packages, costReviews: their consumed attribution reviews }.
