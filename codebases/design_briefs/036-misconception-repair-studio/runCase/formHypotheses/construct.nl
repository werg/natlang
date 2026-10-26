---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMisconceptionHypothesis
---
Construct a MisconceptionHypothesis draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Generate a small set of plausible explanations for the observed pattern, including task ambiguity and accidental error. Identify a question whose answers would distinguish these explanations instead of treating one wrong response as a stable misconception.
