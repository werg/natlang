---
description: releaseTraining domain stage for Cross Cultural Training Adaptation Studio.
args:
  goal: string
  records: AdaptationTest[]
  evidence: EvidenceItem[]
returns: StageResultAdaptedTrainingRelease
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAdaptedTrainingRelease values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Assemble the adapted program with facilitator notes, terminology, and documented limits of applicability. Obtain audience and instructional review, version the relationship to the source program, and avoid presenting one adaptation as universal for an entire culture. The stage gate is: Approved audience scope and preserved learning objectives are explicit.
