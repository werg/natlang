---
description: assessProposals domain stage for Serial Fiction Continuity Room.
args:
  goal: string
  records: StoryObligation[]
  evidence: EvidenceItem[]
returns: StageResultStoryProposalFinding
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftStoryProposalFinding values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Evaluate proposed twists or scenes against chronology, knowledge, motivations, and world rules. Separate genuine contradiction from a reveal that reinterprets prior evidence, and identify what earlier material would need revision to support the new direction. The stage gate is: No unexplained violation of established knowledge or chronology.
