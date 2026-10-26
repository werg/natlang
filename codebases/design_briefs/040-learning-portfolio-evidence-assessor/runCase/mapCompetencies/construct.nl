---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCompetencyMapping
---
Construct a CompetencyMapping draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare evidence with the actual competency definition and performance level. Recognize alternative demonstrations, avoid matching by keywords alone, and identify which facets are supported versus merely mentioned in a reflection.
