---
description: mapParticipants domain stage for Care Transition Logistics Coordinator.
args:
  goal: string
  records: TransitionRequirement[]
  evidence: EvidenceItem[]
returns: StageResultTransitionParticipant
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftTransitionParticipant values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Identify transport providers, equipment suppliers, caregivers, and receiving services with their actual responsibilities. Distinguish availability from acceptance, verify representation and information-sharing authority, and avoid assuming family relationships imply consent or capacity. The stage gate is: Every responsibility has explicit acceptance and permitted information scope.
