---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHabitatObservation
---
Construct a HabitatObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile surveys, field notes, and historical records into a baseline with explicit spatial, temporal, and methodological limits. Distinguish observed absence from unsurveyed areas and preserve rare or uncertain observations for specialist review.
