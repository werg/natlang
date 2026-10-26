---
description: freezeAssembly domain stage for Documentary Assembly Studio.
args:
  goal: string
  records: DocumentaryClearance[]
  evidence: EvidenceItem[]
returns: StageResultDocumentaryRelease
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftDocumentaryRelease values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Produce a versioned edit decision list, script, asset manifest, and unresolved-work register. Verify exact timecodes and duration, bind approvals to the assembly, and invalidate affected checks when a source clip or sequence changes. The stage gate is: All deliverables refer to the same approved assembly and licensed source versions.
