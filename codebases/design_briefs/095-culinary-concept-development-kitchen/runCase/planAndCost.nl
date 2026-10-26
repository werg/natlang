---
description: planAndCost domain stage for Culinary Concept Development Kitchen.
args:
  goal: string
  records: MenuComposition[]
  evidence: EvidenceItem[]
returns: StageResultKitchenProductionPlan
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftKitchenProductionPlan values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Use exact recipe scaling, inventory, costing, and scheduling services to build preparation and service plans. Account for yield, waste, shared equipment, and approved handling requirements; preserve traceability from purchased ingredients to served dishes. The stage gate is: Exact quantities, costs, equipment capacity, and approved handling constraints reconcile.
