---
description: resolveConflicts domain stage for Care Transition Logistics Coordinator.
args:
  goal: string
  records: TransitionTask[]
  evidence: EvidenceItem[]
returns: StageResultTransitionOption
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTransitionOption values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Propose permitted logistical alternatives when providers or times change, explaining consequences for the approved plan. Escalate any alternative that could alter clinical requirements and keep the transition blocked until the appropriate team resolves the conflict. The stage gate is: Administrative replanning never substitutes for clinical approval.
