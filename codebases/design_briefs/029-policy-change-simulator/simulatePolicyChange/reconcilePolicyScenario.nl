---
args: { prior: PolicyScenarioCheckpoint|null, request: PolicyScenarioRequest, dossier: PolicySimulationDossier, delta: PolicyDeltaSet, cohort: CohortReview, replay: PolicyReplayBinding, outcomes: PolicyOutcomeSet|null, regressions: RegressionSuite|null, boundaries: BoundaryMatrix|null, adoption: PolicyGuardrail }
returns: PolicyScenarioCheckpoint
---
Keep each test run bound to exact baseline and candidate policy hashes, cohort case identities, input digest, execution configuration and distinct run IDs. Candidate edits fork new snapshots; old pairs remain historical and are not relabeled. Preserve untested boundaries and indeterminate cohort cases. Rare safeguard regressions remain open until their own review evidence resolves them. Adoption applies only to the tested candidate hash. Duplicate run and adoption receipt replay is a no-op.
