---
description: identifyOptions domain stage for Community Agreement Mediator.
args:
  goal: string
  records: MediationPosition[]
  evidence: EvidenceItem[]
returns: StageResultCommunityOption
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCommunityOption values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate practical alternatives such as scheduling, shared responsibilities, or differentiated use. Preserve each participant’s stated constraints, disclose concessions, and include the option of no agreement rather than manufacturing consensus or pressuring acceptance. The stage gate is: Options are feasible and accurately represent who would give up what.
