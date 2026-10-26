---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MigrationRehearsal[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for coordinateCutover. Freeze the approved mapping and cutover scope, apply final source deltas, and reconcile target results before switching consumers. Track quarantined records and post-cutover corrections explicitly, retaining a safe rollback or forward-repair plan for irreversible changes. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
