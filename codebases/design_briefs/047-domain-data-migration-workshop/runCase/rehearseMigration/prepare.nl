---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationDecision[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for rehearseMigration. Materialize a staged target snapshot, validate referential integrity and business invariants, and compare representative source-target round trips. Rehearse rollback and delta capture so cutover does not lose changes made after the initial extraction. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
