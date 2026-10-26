---
args: { request: SubstituteEvaluationRequest, receipts: EvidencePage[] }
returns: QualificationGuardrail
---
Require an exact conversion receipt with source and target units, dimensional basis, precision and rounding rule whenever catalog values use different units. Keep the original measurements alongside converted values and leave comparison undecided if dimensional equivalence or required tolerance is unknown.
