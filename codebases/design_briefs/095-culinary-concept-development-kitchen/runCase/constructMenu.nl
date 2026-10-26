---
description: constructMenu domain stage for Culinary Concept Development Kitchen.
args:
  goal: string
  records: DishConcept[]
  evidence: EvidenceItem[]
returns: StageResultMenuComposition
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftMenuComposition values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Assemble a balanced menu whose dishes work together across preparation effort, equipment, pacing, and customer choice. Identify shared components and bottlenecks, preserve meaningful alternatives for required diets, and avoid creating variety only through superficial garnish changes. The stage gate is: Menu-level feasibility and required dietary coverage pass.
