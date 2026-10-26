---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFieldObservation
---
Construct a FieldObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Parse field notes and structured submissions while preserving original wording, location precision, and observation method. Distinguish what was directly seen from inferred identity or cause, and flag missing metadata without inventing exact coordinates or times.
