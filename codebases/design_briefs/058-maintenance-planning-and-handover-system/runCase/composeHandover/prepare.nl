---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: MaintenanceProgress[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for composeHandover. Prepare a shift handover describing actual asset state, work in progress, unresolved findings, and required next checks. Require acknowledgment from the receiving owner and preserve the distinction between maintenance completion and authorized return to service. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
