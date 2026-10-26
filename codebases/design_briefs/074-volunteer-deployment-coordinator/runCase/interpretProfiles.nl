---
description: interpretProfiles domain stage for Volunteer Deployment Coordinator.
args:
  goal: string
  records: MissionRequirement[]
  evidence: EvidenceItem[]
returns: StageResultVolunteerCapability
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftVolunteerCapability values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Read volunteer profiles and confirmations to distinguish self-reported skills, verified qualifications, availability, and accepted roles. Preserve limitations and preferences, and avoid treating past participation as consent to a new assignment. The stage gate is: Capability and consent are explicit and current.
