---
description: extractLogistics domain stage for Care Transition Logistics Coordinator.
args:
  goal: string
  records: CareTransition[]
  evidence: EvidenceItem[]
returns: StageResultTransitionRequirement
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTransitionRequirement values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read the approved plan to identify logistical tasks and dependencies, preserving its wording for clinical requirements. Separate administrative coordination from treatment decisions and route any ambiguous clinical instruction back to the authorized team. The stage gate is: No generated change to clinical content or priority.
