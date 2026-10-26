---
description: rebuildRunOfShow domain stage for Live Event Disruption Director.
args:
  goal: string
  records: EventRecoveryOption[]
  evidence: EvidenceItem[]
returns: StageResultEventRunPlan
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEventRunPlan values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Use exact scheduling and resource checks to construct a revised run of show with owners, dependencies, and decision deadlines. Reserve resources before announcing changes, preserve crew readiness, and identify points beyond which a contingency is no longer feasible. The stage gate is: No double booking, impossible setup sequence, or unsupported timing promise.
