---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftInstructionVariant
---
Construct a InstructionVariant draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Rewrite steps for the local context using approved terminology and resources. Preserve sequence, guards, stop conditions, and verification; represent genuine deviations explicitly rather than hiding them in friendlier wording or inferred substitutions.
