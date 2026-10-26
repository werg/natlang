---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: IncidentImpact[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for maintainHypotheses. Generate a bounded set of causal hypotheses from deployment history, dependencies, and symptoms. Track evidence for and against each, choose checks that distinguish mechanisms, and avoid repeatedly confirming the most recent or most fluent explanation. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
