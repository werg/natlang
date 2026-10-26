---
description: constructTeams domain stage for Volunteer Deployment Coordinator.
args:
  goal: string
  records: VolunteerCapability[]
  evidence: EvidenceItem[]
returns: StageResultVolunteerTeam
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftVolunteerTeam values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Build feasible team configurations that collectively cover mission requirements and supervision. Use exact scheduling and capacity checks, explain role assignments, and preserve fair application of supplied assignment principles rather than optimizing convenience alone. The stage gate is: All required capabilities and consent conditions satisfied or the mission remains blocked.
