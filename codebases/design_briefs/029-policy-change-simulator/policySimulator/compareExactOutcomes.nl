---
args: { delta: PolicyDeltaSet, cohort: CohortReview, binding: PolicyReplayBinding }
returns: PolicyOutcomeSet
---
Join only verified run pairs to their frozen case. Compare exact returned outcome, duty, permission, timing and escalation fields. Attribute a changed field to an amendment condition only where its evidence and clause lineage support the causal link; otherwise preserve it as unassigned or unresolved. Never synthesize a run or compute an aggregate from unmatched cases.
