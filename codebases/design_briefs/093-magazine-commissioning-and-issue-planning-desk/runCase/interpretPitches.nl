---
description: interpretPitches domain stage for Magazine Commissioning And Issue Planning Desk.
args:
  goal: string
  records: MagazineIssue[]
  evidence: EvidenceItem[]
returns: StageResultEditorialPitch
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEditorialPitch values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read pitches against the issue brief to identify the actual angle, reporting plan, and audience value. Distinguish an interesting topic from a viable story, preserve uncertainty about access or evidence, and identify overlap with already commissioned work. The stage gate is: Pitch evaluation is grounded in the supplied editorial goals and feasible reporting.
