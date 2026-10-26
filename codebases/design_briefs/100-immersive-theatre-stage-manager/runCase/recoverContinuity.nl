---
description: recoverContinuity domain stage for Immersive Theatre Stage Manager.
args:
  goal: string
  records: TheatreCuePlan[]
  evidence: EvidenceItem[]
returns: StageResultTheatreRecovery
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTheatreRecovery values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: When cast, props, or audience paths change, propose bounded in-character recovery material and feasible scene adjustments. Preserve revealed facts and participant boundaries, obtain stage-manager approval, and log the actual path for subsequent scenes and post-show reset. The stage gate is: Recovery preserves safety, consent, timing feasibility, and narrative continuity.
