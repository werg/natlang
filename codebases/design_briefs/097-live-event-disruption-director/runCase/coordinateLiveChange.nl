---
description: coordinateLiveChange domain stage for Live Event Disruption Director.
args:
  goal: string
  records: EventRunPlan[]
  evidence: EvidenceItem[]
returns: StageResultEventChange
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEventChange values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Obtain producer approval, issue role-specific updates, and confirm that critical staff received the same revision. Track actual execution and adapt only affected elements as new events arrive, preserving an authoritative current run plan. The stage gate is: Safety authority, staff acknowledgment, and public communications align with the current plan.
