---
description: coordinatePerformance domain stage for Immersive Theatre Stage Manager.
args:
  goal: string
  records: SceneVariant[]
  evidence: EvidenceItem[]
returns: StageResultTheatreCuePlan
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTheatreCuePlan values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Build an exact cue and movement plan using approved production constraints, then provide role-specific briefings. Confirm critical acknowledgments, preserve stage-manager control, and distinguish a proposed cue from an executed cue in the live state. The stage gate is: No conflicting resource use, impossible timing, or unauthorized safety-critical cue.
