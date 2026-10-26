---
description: generateDirections domain stage for Brand And Design Language Studio.
args:
  goal: string
  records: BrandConstraint[]
  evidence: EvidenceItem[]
returns: StageResultCreativeDirection
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCreativeDirection values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Develop materially different creative directions with coherent naming, tone, and message logic. Explain what each direction emphasizes and excludes, avoid cosmetic variants of one idea, and keep unsupported product promises outside the creative system. The stage gate is: Directions satisfy factual constraints and are meaningfully distinct.
