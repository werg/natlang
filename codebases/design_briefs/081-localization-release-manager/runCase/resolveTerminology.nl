---
description: resolveTerminology domain stage for Localization Release Manager.
args:
  goal: string
  records: LocalizationSegment[]
  evidence: EvidenceItem[]
returns: StageResultLocaleTermDecision
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftLocaleTermDecision values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Match terms to concepts and approved locale guidance, considering polysemy, grammatical role, and product meaning. Preserve scoped exceptions and route genuine terminology conflicts to the owner rather than choosing whichever translation appears most often. The stage gate is: Term choice is concept- and context-specific, with versioned authority.
