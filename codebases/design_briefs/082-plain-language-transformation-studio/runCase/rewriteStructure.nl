---
description: rewriteStructure domain stage for Plain Language Transformation Studio.
args:
  goal: string
  records: ReaderTask[]
  evidence: EvidenceItem[]
returns: StageResultPlainLanguageDraft
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPlainLanguageDraft values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reorganize information around reader tasks, use clearer sentences, and introduce explanations or examples where they help. Preserve technical terms when necessary, label examples as illustrative, and avoid creating new rules through oversimplified summaries. The stage gate is: Rewrite preserves every material condition, exception, and uncertainty.
