---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IncidentHypothesis[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for selectDiagnostics. Choose from approved read-only or bounded diagnostic operations, considering expected discrimination and operational risk. Validate scope and rate limits, request commander approval for intrusive checks, and record what each possible result would change. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
