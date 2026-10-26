---
description: replayLearning domain stage for Cross Cultural Training Adaptation Studio.
args:
  goal: string
  records: AdaptedScenario[]
  evidence: EvidenceItem[]
returns: StageResultAdaptationTest
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAdaptationTest values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare original and adapted scenarios through representative correct, incorrect, and borderline responses. Detect shifts in difficulty or meaning, inspect whether examples imply unsupported norms, and revise only the elements responsible for the drift. The stage gate is: The same competency distinctions remain observable in both versions.
