---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: AnalysisMeaning[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for inspectData. Inspect schemas, dictionaries, representative records, and access constraints to determine whether the question is answerable. Distinguish absent data from inaccessible data and identify proxy fields that would change the meaning rather than silently substituting them. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
