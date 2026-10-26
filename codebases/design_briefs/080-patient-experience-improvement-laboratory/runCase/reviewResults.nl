---
description: reviewResults domain stage for Patient Experience Improvement Laboratory.
args:
  goal: string
  records: ServiceExperiment[]
  evidence: EvidenceItem[]
returns: StageResultExperienceOutcome
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftExperienceOutcome values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Reconcile computed administrative metrics with follow-up feedback and implementation evidence. Check changed denominators, observation channels, and minority experiences, then recommend next actions without claiming clinical benefit beyond the study’s scope. The stage gate is: Reported improvements match exact measures and preserve limits and counterevidence.
