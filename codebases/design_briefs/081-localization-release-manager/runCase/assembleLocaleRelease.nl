---
description: assembleLocaleRelease domain stage for Localization Release Manager.
args:
  goal: string
  records: LocaleQualityFinding[]
  evidence: EvidenceItem[]
returns: StageResultLocaleReleasePacket
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftLocaleReleasePacket values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Build a locale bundle bound to source, glossary, context, and review versions. Identify stale translations after source changes, apply explicit fallback rules, and release only approved segments without mixing incompatible editions in one user journey. The stage gate is: Bundle is internally version-consistent and every fallback is declared and safe.
