---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MaintenanceProcedureFit[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for constructWorkPlan. Decompose the approved procedure into accountable tasks, resource reservations, and verification steps. Sequence shared access and parts availability, preserve stop conditions, and distinguish planned work from authorized readiness to begin. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
