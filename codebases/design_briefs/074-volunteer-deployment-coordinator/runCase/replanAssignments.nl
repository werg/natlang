---
description: replanAssignments domain stage for Volunteer Deployment Coordinator.
args:
  goal: string
  records: VolunteerBriefing[]
  evidence: EvidenceItem[]
returns: StageResultVolunteerRevision
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftVolunteerRevision values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Process withdrawals and changed mission conditions by identifying exact capability gaps. Propose replacement teams or reduced scope, seek fresh consent for changed roles, and cancel unsafe or incomplete deployments rather than pretending nominal headcount is sufficient. The stage gate is: Replanned teams retain required coverage, supervision, and current consent.
