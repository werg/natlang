---
args: { request: PolicyScenarioRequest, delta: PolicyDeltaSet, boundaries: BoundaryMatrix, regressions: RegressionSuite }
returns: PolicyOptionProposal
---
Propose a policy option only from a version-bound delta and tested boundaries. State affected cohort, observed changes, retained safeguards, unresolved tradeoffs, and owner review required for each regression. Any adoption proposal pins the tested candidate hash and requires rerun after later edits.
