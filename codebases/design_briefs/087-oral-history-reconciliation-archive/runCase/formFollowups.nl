---
description: formFollowups domain stage for Oral History Reconciliation Archive.
args:
  goal: string
  records: RecollectionComparison[]
  evidence: EvidenceItem[]
returns: StageResultOralHistoryQuestion
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftOralHistoryQuestion values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate neutral follow-up questions that could clarify identities, chronology, or context without leading speakers toward a preferred account. Respect consent and participation boundaries, and distinguish clarification from challenging a person’s memory as false. The stage gate is: Questions are nonleading, relevant, and within consent scope.
