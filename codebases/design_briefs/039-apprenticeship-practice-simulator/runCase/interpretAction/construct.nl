---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftLearnerAction
---
Construct a LearnerAction draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret the learner’s natural-language response against the available action set. Distinguish a proposed action from an executed one, ask for clarification when ambiguity affects consequences, and avoid rewarding jargon without the corresponding operational behavior.
