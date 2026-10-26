---
description: modelCharacterViews domain stage for Persistent World Game Director.
args:
  goal: string
  records: PlayerIntent[]
  evidence: EvidenceItem[]
returns: StageResultCharacterView
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftCharacterView values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Build each character’s view from witnessed events, received messages, memory, and approved world facts. Separate true facts from beliefs and secrets, preserve obligations and motivations, and prevent the narrator’s global knowledge from leaking into character decisions. The stage gate is: Character-visible context contains only authorized knowledge.
