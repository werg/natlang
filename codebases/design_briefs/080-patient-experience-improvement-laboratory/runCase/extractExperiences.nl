---
description: extractExperiences domain stage for Patient Experience Improvement Laboratory.
args:
  goal: string
  records: ExperienceStudy[]
  evidence: EvidenceItem[]
returns: StageResultExperienceEpisode
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExperienceEpisode values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Process surveys, complaints, and calls to identify administrative experiences and their context. Separate access or communication issues from clinical conclusions, preserve mixed experiences, and avoid inferring diagnoses or treatment quality from administrative feedback. The stage gate is: Findings stay within observed administrative experience.
