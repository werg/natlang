---
description: applyChannels domain stage for Brand And Design Language Studio.
args:
  goal: string
  records: BrandSystem[]
  evidence: EvidenceItem[]
returns: StageResultBrandAsset
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftBrandAsset values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate channel-specific assets that serve actual user tasks while respecting length, format, and factual constraints. Adapt emphasis and tone to context, preserve required disclosures, and avoid treating a slogan as a substitute for useful product information. The stage gate is: Each asset is useful in its channel and consistent with the approved system.
