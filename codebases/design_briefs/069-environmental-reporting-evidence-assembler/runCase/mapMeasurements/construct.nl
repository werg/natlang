---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftEnvironmentalMeasure
---
Construct a EnvironmentalMeasure draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile measurements, invoices, contractor statements, and field records with their actual methods and periods. Preserve estimates versus direct measurements, identify missing coverage, and retain original units and uncertainty for exact conversion services.
