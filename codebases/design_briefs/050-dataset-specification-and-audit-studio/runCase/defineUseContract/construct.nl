---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDatasetUse
---
Construct a DatasetUse draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret the intended application and decisions the dataset will support. Define target population, task boundaries, prohibited uses, and meaningful evaluation slices; avoid treating easily available data as representative of the intended use.
