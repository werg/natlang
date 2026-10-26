---
description: constructTermOptions domain stage for Multilingual Terminology Negotiation Workbench.
args:
  goal: string
  records: MultilingualConcept[]
  evidence: EvidenceItem[]
returns: StageResultTermOption
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTermOption values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Propose terms and scoped usage rules considering approved domain conventions, grammatical forms, and audience needs. Explain tradeoffs among clarity, continuity, and precision, and retain regional or contextual variants where one universal term would change meaning. The stage gate is: Each option preserves the intended concept in its stated context.
