---
description: routeQuestions domain stage for Appointment Preparation Service.
args:
  goal: string
  records: PreparationChecklist[]
  evidence: EvidenceItem[]
returns: StageResultPreparationQuestion
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPreparationQuestion values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Form focused questions for the appropriate staff when instructions conflict with practical circumstances or remain unclear. Track authoritative responses, avoid answering medical questions from general model knowledge, and update the checklist only from approved clarification. The stage gate is: Clinical clarification comes from an authorized source.
