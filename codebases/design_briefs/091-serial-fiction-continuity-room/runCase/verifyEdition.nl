---
description: verifyEdition domain stage for Serial Fiction Continuity Room.
args:
  goal: string
  records: StoryRevision[]
  evidence: EvidenceItem[]
returns: StageResultStoryEdition
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftStoryEdition values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Re-run continuity checks over affected scenes and downstream knowledge states, inspect arc payoffs, and assemble a coherent editorial diff. Record approved canon changes and preserve prior editions so future work uses the correct story-world state. The stage gate is: Global continuity and approved narrative constraints pass; creative quality receives separate editorial review.
