---
description: maintainContinuity domain stage for Persistent World Game Director.
args:
  goal: string
  records: WorldConsequence[]
  evidence: EvidenceItem[]
returns: StageResultWorldContinuity
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftWorldContinuity values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Review long-term consistency and introduce events that develop existing tensions or obligations. Check that new events are possible from current state, respect participant boundaries, and preserve stable checkpoints for replay, moderation, and recovery. The stage gate is: No retroactive state invention or unapproved content-boundary change.
