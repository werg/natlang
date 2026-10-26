---
description: negotiateDecision domain stage for Multilingual Terminology Negotiation Workbench.
args:
  goal: string
  records: TermOption[]
  evidence: EvidenceItem[]
returns: StageResultTerminologyDecision
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTerminologyDecision values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Present alternatives and concrete usage examples to the relevant experts, reconcile their comments, and distinguish stylistic preference from conceptual disagreement. Record scoped acceptance and unresolved dissent rather than treating a majority vote as proof of semantic equivalence. The stage gate is: Decision authority and applicability are explicit; unresolved conceptual conflict remains visible.
