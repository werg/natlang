---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDataSuitability
---
Construct a DataSuitability draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Inspect schemas, dictionaries, representative records, and access constraints to determine whether the question is answerable. Distinguish absent data from inaccessible data and identify proxy fields that would change the meaning rather than silently substituting them.
