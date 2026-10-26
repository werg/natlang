---
description: writeArtifacts domain stage for Escape Room Foundry.
args:
  goal: string
  records: PuzzleNode[]
  evidence: EvidenceItem[]
returns: StageResultPuzzleArtifact
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPuzzleArtifact values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Create in-world documents, dialogue, labels, and clues that support the intended reasoning without relying on designer-only knowledge. Generate plausible player misreadings, preserve thematic coherence, and distinguish fair misdirection from ambiguity that makes the puzzle underdetermined. The stage gate is: Clues are sufficient, internally consistent, and do not require inaccessible external knowledge.
