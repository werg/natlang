---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAnalysisAnswer
---
Construct a AnalysisAnswer draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Formulate the result using exact computed values and the approved question interpretation. Explain exclusions, uncertainty, and unanswerable components; attach the executable work order and lineage so another analyst can reproduce or revise the analysis.
