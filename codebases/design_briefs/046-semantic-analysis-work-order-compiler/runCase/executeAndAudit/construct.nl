---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAnalysisSlice
---
Construct a AnalysisSlice draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Execute finite partitions, cache semantic judgments by record and rubric version, and reconcile every input disposition. Audit sampled positive and negative labels, exact join cardinalities, temporal boundaries, and denominators before materializing the final aggregate.
