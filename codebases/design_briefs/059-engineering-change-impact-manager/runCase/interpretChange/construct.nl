---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDesignDelta
---
Construct a DesignDelta draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare approved design revisions and change rationale to identify functional, interface, and process effects. Separate editorial drawing updates from meaningful changes and preserve requirements that remain binding despite a local modification.
