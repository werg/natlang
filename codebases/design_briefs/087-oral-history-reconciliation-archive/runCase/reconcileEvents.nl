---
description: reconcileEvents domain stage for Oral History Reconciliation Archive.
args:
  goal: string
  records: OralHistoryReference[]
  evidence: EvidenceItem[]
returns: StageResultRecollectionComparison
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftRecollectionComparison values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare accounts of potentially shared events, aligning chronology and scope while preserving different perspectives. Distinguish factual contradiction from differences in emphasis or memory precision, and do not manufacture one authoritative narrative from majority agreement. The stage gate is: Conflicting recollections remain attributed and searchable.
