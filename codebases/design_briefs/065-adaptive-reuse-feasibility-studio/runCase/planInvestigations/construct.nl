---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftReuseInvestigation
---
Construct a ReuseInvestigation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Identify the surveys, specialist reviews, or evidence needed to distinguish feasible concepts. Prioritize investigations by their ability to change a decision, and specify how positive, negative, or inconclusive findings would alter the concept.
