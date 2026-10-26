---
description: verifyDelivery domain stage for Health Record Request Fulfillment Workbench.
args:
  goal: string
  records: RecordFulfillment[]
  evidence: EvidenceItem[]
returns: StageResultRecordDelivery
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRecordDelivery values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Recheck authorization immediately before recipient-bound delivery, record actual receipt where available, and track partial fulfillment or corrections. Preserve an audit trail without exposing sensitive record bodies in operational logs or notifications. The stage gate is: Correct recipient, current authority, and exact released artifact identity are verified.
