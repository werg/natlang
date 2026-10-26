---
description: prepareBriefings domain stage for Volunteer Deployment Coordinator.
args:
  goal: string
  records: VolunteerTeam[]
  evidence: EvidenceItem[]
returns: StageResultVolunteerBriefing
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftVolunteerBriefing values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate role-specific briefings with task boundaries, practical logistics, and escalation routes. Share only necessary participant information, distinguish instructions from optional suggestions, and obtain acknowledgment for consequential responsibilities before deployment. The stage gate is: Briefings preserve approved procedures and contain no unauthorized personal disclosures.
