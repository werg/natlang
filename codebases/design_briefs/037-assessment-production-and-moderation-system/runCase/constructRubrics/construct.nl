---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftScoringRubric
---
Construct a ScoringRubric draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Define observable scoring criteria and partial-credit boundaries, separating conceptual correctness from presentation preferences. Include correct alternative methods and common errors, and state how ambiguity or missing evidence should be handled.
