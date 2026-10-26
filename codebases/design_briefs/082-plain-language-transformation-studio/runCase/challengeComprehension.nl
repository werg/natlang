---
description: challengeComprehension domain stage for Plain Language Transformation Studio.
args:
  goal: string
  records: PlainLanguageDraft[]
  evidence: EvidenceItem[]
returns: StageResultComprehensionProbe
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftComprehensionProbe values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Create realistic comprehension questions and boundary scenarios that test whether readers would act correctly. Include cases involving exceptions and negative requirements, then compare source and rewrite answers to detect semantic drift. The stage gate is: Source and rewrite imply the same consequential answers.
