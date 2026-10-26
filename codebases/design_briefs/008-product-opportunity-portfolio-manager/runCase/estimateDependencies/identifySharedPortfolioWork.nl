---
args: { packages: WorkPackage[], costReviews: WorkPackageCostReview[], evidence: EvidencePage[] }
returns: SharedWorkAnalysis
---
For each pair of packages that claims the same enabling work or user benefit call compareSharedDependency(left, right, costReviews, evidence). Use the actual operational-overlap judgment, dependency IDs, evidence and distinct-outcome risks. Cluster only supported same-work dependencies; keep similar labels with different workflows separate. Set countOnce only for a verified shared cost or duplicated benefit. Return { clusters: supported shared-work clusters, comparisons: all pair judgments including unresolved and distinct pairs }.
