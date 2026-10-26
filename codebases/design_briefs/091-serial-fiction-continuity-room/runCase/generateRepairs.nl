---
description: generateRepairs domain stage for Serial Fiction Continuity Room.
args:
  goal: string
  records: StoryProposalFinding[]
  evidence: EvidenceItem[]
returns: StageResultStoryRevision
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftStoryRevision values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Generate distinct repairs such as changing the new scene, seeding earlier evidence, or revising an approved premise through explicit editorial choice. Preserve voice and emotional intent, and explain narrative tradeoffs rather than selecting the easiest continuity patch. The stage gate is: Repairs satisfy hard canon constraints unless a reviewed canon change is explicitly proposed.
