---
description: segmentWithContext domain stage for Localization Release Manager.
args:
  goal: string
  records: LocalizationRelease[]
  evidence: EvidenceItem[]
returns: StageResultLocalizationSegment
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftLocalizationSegment values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Parse source content into translatable units while preserving structured markup, placeholders, plurals, and surrounding context. Distinguish identical strings used for different purposes and attach the product state or audience needed to choose the correct meaning. The stage gate is: Exact token and markup structure preserved; context is not discarded during deduplication.
