---
description: reviewTastings domain stage for Culinary Concept Development Kitchen.
args:
  goal: string
  records: KitchenProductionPlan[]
  evidence: EvidenceItem[]
returns: StageResultCulinaryRevision
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCulinaryRevision values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Interpret structured tasting and service rehearsal notes to identify creative and practical improvements. Distinguish subjective preference from repeatable defects, revise the smallest relevant components, and revalidate cost, dietary information, and service dependencies after every change. The stage gate is: Revised menu passes hard checks; culinary quality remains chef-reviewed rather than self-certified.
