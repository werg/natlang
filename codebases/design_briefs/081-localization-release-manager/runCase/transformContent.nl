---
description: transformContent domain stage for Localization Release Manager.
args:
  goal: string
  records: LocaleTermDecision[]
  evidence: EvidenceItem[]
returns: StageResultLocalizedSegment
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftLocalizedSegment values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Translate or adapt according to communicative purpose while preserving product facts, obligations, and constraints. Generate alternatives for culturally or linguistically sensitive phrasing using supplied guidance, and keep unresolved meaning questions visible rather than hiding them in fluent text. The stage gate is: Meaning, protected tokens, and required product terminology survive transformation.
