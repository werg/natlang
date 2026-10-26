---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftAnalysisWorkGraph
---
Construct a AnalysisWorkGraph draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Construct a typed DAG of exact filters, joins, bounded semantic classification, temporal matching, and aggregation. Push exact restrictive filters before semantic work when meaning is preserved; attach provenance and missing-data rules to every edge and reject unsupported operations.
