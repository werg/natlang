---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFacilityResolution
---
Construct a FacilityResolution draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare technician completion evidence and subsequent occupant observations with the original complaint. Distinguish a replaced component from resolution of the symptom, detect recurrence, and reopen only the relevant issue when work does not achieve the intended outcome.
