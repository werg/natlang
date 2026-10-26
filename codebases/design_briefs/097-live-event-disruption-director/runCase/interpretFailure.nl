---
description: interpretFailure domain stage for Live Event Disruption Director.
args:
  goal: string
  records: EventCommitment[]
  evidence: EvidenceItem[]
returns: StageResultEventDisruption
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEventDisruption values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read staff reports and provider updates to determine what capability is lost and which parts of the program are affected. Separate confirmed facts from estimates and avoid assuming a missing person or resource will return by a convenient time. The stage gate is: Disruption scope and uncertainty are explicit.
