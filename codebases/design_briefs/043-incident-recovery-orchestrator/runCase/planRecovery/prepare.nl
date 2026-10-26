---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DiagnosticAction[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for planRecovery. Select a recovery path from approved runbooks using current evidence and legal action options. Sequence reversible actions first where appropriate, preserve prerequisites, and avoid improvising destructive commands or treating a successful command as recovered service. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
