---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftResponseObservation
---
Construct a ResponseObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Inspect the learner’s response against the task and accepted solution space. Distinguish conceptual errors, arithmetic slips, misunderstood wording, and incomplete work; preserve alternative correct strategies rather than forcing one canonical explanation.
