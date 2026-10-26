---
description: constructSequence domain stage for Care Transition Logistics Coordinator.
args:
  goal: string
  records: TransitionParticipant[]
  evidence: EvidenceItem[]
returns: StageResultTransitionTask
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTransitionTask values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Build an exact dependency plan for equipment, access, transport, appointments, and acknowledgments. Identify conditions that must be ready before transfer and expose impossible timing or missing ownership rather than quietly relaxing the approved plan. The stage gate is: All logistical prerequisites are feasible and traceable to the approved plan.
