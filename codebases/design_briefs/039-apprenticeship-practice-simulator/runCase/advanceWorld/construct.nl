---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPracticeTransition
---
Construct a PracticeTransition draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Apply the exact simulator’s legal transition rules, then formulate context-appropriate responses from actors who know only their own information. Preserve consequences across turns and do not reveal hidden facts merely to help the learner succeed.
