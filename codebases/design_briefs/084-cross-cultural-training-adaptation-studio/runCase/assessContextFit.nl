---
description: assessContextFit domain stage for Cross Cultural Training Adaptation Studio.
args:
  goal: string
  records: TrainingMechanism[]
  evidence: EvidenceItem[]
returns: StageResultContextFitFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftContextFitFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Compare scenario elements with supplied audience guidance and reviewer input. Identify unfamiliar references, inappropriate assumptions, or practical mismatches without inferring cultural traits from nationality, ethnicity, or other sensitive attributes. The stage gate is: Adaptation rationale is evidence-based and audience-specific, not stereotyped.
