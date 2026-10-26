---
description: modelProduction domain stage for Immersive Theatre Stage Manager.
args:
  goal: string
  records: TheatreScene[]
  evidence: EvidenceItem[]
returns: StageResultTheatreConstraint
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTheatreConstraint values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reconcile cast availability, venue constraints, props, accessibility arrangements, and participant boundaries. Identify scenes requiring qualified stage control or specific resources, and keep physical-safety instructions outside improvisational model authority. The stage gate is: Required resources and consent conditions are explicit and current.
