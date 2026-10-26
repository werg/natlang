---
description: interpretAuthorization domain stage for Health Record Request Fulfillment Workbench.
args:
  goal: string
  records: HealthRecordRequest[]
  evidence: EvidenceItem[]
returns: StageResultRecordAuthorization
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRecordAuthorization values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the request and authorization together to determine recipient, purpose, categories, dates, and any restrictions. Distinguish a requested scope from an authorized scope and route ambiguous authority to the records officer before retrieval. The stage gate is: Current authorization covers the exact recipient and proposed disclosure purpose.
