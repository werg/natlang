---
description: generateAlternatives domain stage for Cross Cultural Training Adaptation Studio.
args:
  goal: string
  records: ContextFitFinding[]
  evidence: EvidenceItem[]
returns: StageResultAdaptedScenario
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftAdaptedScenario values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Invent alternative settings, examples, and interactions that remain plausible for the supplied context. Preserve the same relevant ambiguity, constraints, and skill demand, and avoid making the adapted task easier or changing the intended lesson unintentionally. The stage gate is: Instructional equivalence and contextual plausibility are separately reviewed.
