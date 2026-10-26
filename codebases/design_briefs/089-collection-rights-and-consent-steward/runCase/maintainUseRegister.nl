---
description: maintainUseRegister domain stage for Collection Rights And Consent Steward.
args:
  goal: string
  records: ClearanceTask[]
  evidence: EvidenceItem[]
returns: StageResultCollectionUseRecord
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCollectionUseRecord values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Record approved uses, required attribution, expiry, and downstream distribution. Recheck conditions before release, schedule removals or renewals where needed, and propagate consent changes to affected derivatives and public surfaces. The stage gate is: Approved use, actual distribution, attribution, and duration remain aligned.
