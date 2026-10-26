---
description: recoverMeaning domain stage for Plain Language Transformation Studio.
args:
  goal: string
  records: PlainLanguageProject[]
  evidence: EvidenceItem[]
returns: StageResultMeaningUnit
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftMeaningUnit values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Parse the source into claims, duties, permissions, conditions, and exceptions before rewriting. Identify terms that require explanation and preserve uncertainty or scope; do not treat dense wording as permission to remove consequential distinctions. The stage gate is: All operative meaning units are represented with source locations.
