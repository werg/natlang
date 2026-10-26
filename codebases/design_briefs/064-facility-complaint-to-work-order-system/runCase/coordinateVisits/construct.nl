---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFacilityVisit
---
Construct a FacilityVisit draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Schedule visits using exact availability and resource constraints, obtain required access consent, and coordinate multiple trades. Preserve cancellations and failed-access events, and update occupants with truthful windows rather than unsupported precise arrival promises.
