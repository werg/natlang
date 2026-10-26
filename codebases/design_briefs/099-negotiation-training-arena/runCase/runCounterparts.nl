---
description: runCounterparts domain stage for Negotiation Training Arena.
args:
  goal: string
  records: TrainingOffer[]
  evidence: EvidenceItem[]
returns: StageResultCounterpartAction
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCounterpartAction values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Choose simulated counterpart responses from permitted actions using only that role’s information and preferences. Adapt strategy to the interaction without violating conduct rules, inventing authority, or exposing hidden utility values to the learner. The stage gate is: Role knowledge, mandate, and conduct constraints hold throughout the episode.
