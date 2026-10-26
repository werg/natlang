---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftObservationLink
---
Construct a ObservationLink draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare observer, time, location, media hashes, and narrative context to identify repeated submissions of one event. Preserve independent observations of the same phenomenon and avoid merging nearby events solely because they share a likely identity.
