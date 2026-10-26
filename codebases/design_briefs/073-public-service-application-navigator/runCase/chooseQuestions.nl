---
description: chooseQuestions domain stage for Public Service Application Navigator.
args:
  goal: string
  records: ServicePathway[]
  evidence: EvidenceItem[]
returns: StageResultNavigationQuestion
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftNavigationQuestion values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Select the next question by how much it resolves relevant pathway uncertainty while minimizing burden and sensitivity. Reuse authorized existing evidence, explain why the information is needed, and allow the person to decline or report that it is unknown. The stage gate is: No unnecessary collection or repeated request for already available evidence.
