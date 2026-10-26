---
description: linkJourneyEvents domain stage for Patient Experience Improvement Laboratory.
args:
  goal: string
  records: ExperienceEpisode[]
  evidence: EvidenceItem[]
returns: StageResultJourneyLink
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftJourneyLink values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Relate feedback to appointment, contact, and referral events using authorized identifiers and time windows. Distinguish repeated reports of one episode from independent experiences, and preserve uncertain linkage rather than exposing unrelated patient records. The stage gate is: Patient isolation and evidence-backed episode linkage.
