---
description: selectSceneVariants domain stage for Immersive Theatre Stage Manager.
args:
  goal: string
  records: TheatreConstraint[]
  evidence: EvidenceItem[]
returns: StageResultSceneVariant
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftSceneVariant values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Select or generate approved-style variants that fit audience choices and current resources. Preserve the dramatic beat, character knowledge, and consent boundaries; avoid replacing a missing prop with an unsafe physical action or unsupported audience involvement. The stage gate is: Variant remains narratively coherent and operationally permitted.
