---
args: { outcomes: PolicyOutcomeSet, regressions: RegressionSuite, cohort: CohortReview }
returns: BoundaryMatrix
---
Describe tested boundaries using exact paired cases and their replay results. Keep combinations absent from the cohort untested, missing-field cases indeterminate, and regressions visible even when aggregate counts favor the candidate. Do not extrapolate beyond tested boundaries.
