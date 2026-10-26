---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFacilityWorkOrder
---
Construct a FacilityWorkOrder draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Choose approved inspection or repair task types and required trade qualifications. Sequence dependencies, include access arrangements and occupant impact, and route safety-sensitive or uncertain cases to qualified personnel rather than generating improvised repairs.
