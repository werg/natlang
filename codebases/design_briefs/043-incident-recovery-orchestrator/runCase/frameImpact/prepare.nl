---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for frameImpact. Combine alerts, user reports, and service telemetry to describe observed impact and uncertainty. Separate symptoms from causes and distinguish missing telemetry from healthy behavior; identify the service outcomes that recovery must actually restore. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
