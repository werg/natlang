---
description: verifyReadiness domain stage for Care Transition Logistics Coordinator.
args:
  goal: string
  records: TransitionOption[]
  evidence: EvidenceItem[]
returns: StageResultTransitionReadiness
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTransitionReadiness values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Recheck actual delivery, provider acceptance, access, and participant acknowledgments before the transition. Distinguish scheduled from completed prerequisites and record the authorized release decision separately from the program’s readiness summary. The stage gate is: No transition-ready claim while a required logistical prerequisite is unverified.
