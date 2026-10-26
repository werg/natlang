---
description: locateRecords domain stage for Health Record Request Fulfillment Workbench.
args:
  goal: string
  records: RecordAuthorization[]
  evidence: EvidenceItem[]
returns: StageResultRecordSelection
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRecordSelection values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Search authorized inventories using verified patient and episode identifiers, preserving date and category boundaries. Distinguish unavailable records from records outside scope and prevent similar identities or merged historical accounts from contaminating the packet. The stage gate is: Correct patient identity and explicit coverage manifest.
