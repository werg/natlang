---
description: recoverEventPromises domain stage for Live Event Disruption Director.
args:
  goal: string
  records: LiveEvent[]
  evidence: EvidenceItem[]
returns: StageResultEventCommitment
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftEventCommitment values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reconcile agendas, contracts, accessibility needs, and participant communications to identify commitments that matter. Distinguish published promises from internal preferences and preserve dependencies such as travel, technical setup, and required access arrangements. The stage gate is: Commitment priority follows supplied authority and actual promises.
