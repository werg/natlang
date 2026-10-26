---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftModerationDecision
---
Construct a ModerationDecision draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare scorer disagreements and appeals against the released rubric and actual response. Identify systematic rubric defects, propose scoped corrections, and apply approved changes consistently to all affected learners rather than only those who appealed.
