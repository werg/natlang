---
description: qualifyIssue domain stage for Magazine Commissioning And Issue Planning Desk.
args:
  goal: string
  records: EditorialTask[]
  evidence: EvidenceItem[]
returns: StageResultIssueReadiness
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftIssueReadiness values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reconcile final pieces, captions, headlines, credits, layout, and publication timing as one issue. Check that promotional language does not overstate article claims, preserve contingency plans for late withdrawals, and freeze the approved edition. The stage gate is: Complete issue meets scope, rights, factual, and production gates.
