---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftFacilityIssueGroup
---
Construct a FacilityIssueGroup draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare reports by location, asset dependencies, timing, and symptom mechanism. Group likely shared issues for coordinated investigation while retaining unrelated complaints that use similar language; do not close a new report solely because a nearby task was completed.
