---
description: mapPositions domain stage for Public Consultation Synthesis Studio.
args:
  goal: string
  records: SubmissionRelation[]
  evidence: EvidenceItem[]
returns: StageResultConsultationTheme
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftConsultationTheme values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Group arguments by the actual concern and reasoning, not only topic words. Preserve minority and cross-cutting positions, identify disagreements about facts versus values, and represent tradeoffs without averaging incompatible views into false consensus. The stage gate is: Every substantive argument has a theme or explicit out-of-scope disposition.
