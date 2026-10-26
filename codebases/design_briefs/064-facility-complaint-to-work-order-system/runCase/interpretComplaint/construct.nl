---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFacilityObservation
---
Construct a FacilityObservation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read occupant descriptions to identify location, symptom, timing, and practical impact. Distinguish observed behavior from guessed causes, resolve ambiguous room names, and preserve access or communication preferences without inferring sensitive personal circumstances.
