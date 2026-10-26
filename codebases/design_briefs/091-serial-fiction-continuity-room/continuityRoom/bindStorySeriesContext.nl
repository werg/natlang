---
args: { series: StorySeries; sourceEvidence: EvidenceItem[]; edition: SerialEdition; change: StoryChange }
returns: BoundStoryContext
---
Use the StorySeries top-level fields `seriesId`, `manuscriptIds`, `canonVersion`, `editorId`, and `creativeBrief` as active-series constraints. Do not read those facts from the generic fields map. Bind the edition identity to seriesId and its canon revision to canonVersion while retaining the supplied approval state, scenes, knowledge events, obligations and exact edition evidence. Bind the author's request context to creativeBrief and retain the changed scene proposal as a draft against that canon; do not call the proposal itself approved. Verify every series.sourceRefs and series.evidenceIds entry against sourceEvidence by exact ID and version. Return evidence gaps separately, and leave the supplied checkpoint version unchanged so stale checkpoints remain detectable.
