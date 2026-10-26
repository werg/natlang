---
description: interpretBrief domain stage for Culinary Concept Development Kitchen.
args:
  goal: string
  records: MenuProject[]
  evidence: EvidenceItem[]
returns: StageResultMenuRequirement
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftMenuRequirement values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Translate the culinary brief into desired experience, service style, supplied dietary requirements, equipment, and budget constraints. Distinguish preferences from mandatory restrictions and use only verified ingredient and allergen information rather than inferring safety from a dish name. The stage gate is: Mandatory dietary and operational constraints are explicit and source-backed.
