---
description: interpretMission domain stage for Volunteer Deployment Coordinator.
args:
  goal: string
  records: VolunteerMission[]
  evidence: EvidenceItem[]
returns: StageResultMissionRequirement
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftMissionRequirement values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Translate the mission brief into concrete tasks, capability requirements, and practical constraints. Distinguish essential qualifications from preferences, identify supervision needs, and preserve safety or safeguarding requirements supplied by the organization. The stage gate is: Requirements are task-based and not inferred from volunteer demographics.
