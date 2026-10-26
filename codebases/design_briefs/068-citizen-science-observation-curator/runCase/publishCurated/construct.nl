---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCuratedObservation
---
Construct a CuratedObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Prepare a curated record with confidence categories grounded in evidence, preserved alternatives, and project-specific disclosure controls. Publish provisional and verified statuses distinctly and update derivatives when experts or observers provide corrections.
