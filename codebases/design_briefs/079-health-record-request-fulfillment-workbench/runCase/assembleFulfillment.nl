---
description: assembleFulfillment domain stage for Health Record Request Fulfillment Workbench.
args:
  goal: string
  records: RecordDisclosureFinding[]
  evidence: EvidenceItem[]
returns: StageResultRecordFulfillment
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRecordFulfillment values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Build a complete or explicitly partial packet with a manifest and explanation of outstanding items. Apply approved redactions through trusted rendering services, verify extracted content and metadata, and bind approval to exact artifact hashes. The stage gate is: No recoverable restricted content, missing unexplained item, or mismatched patient document.
