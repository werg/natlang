---
description: reviewCoherence domain stage for Brand And Design Language Studio.
args:
  goal: string
  records: BrandAsset[]
  evidence: EvidenceItem[]
returns: StageResultBrandReview
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftBrandReview values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare the asset collection for conceptual, terminological, and factual coherence. Separate hard defects from subjective creative preference, obtain client review of meaningful alternatives, and version the system and dependent assets together. The stage gate is: No unsupported claim or unresolved hard inconsistency; originality receives separate human evaluation.
