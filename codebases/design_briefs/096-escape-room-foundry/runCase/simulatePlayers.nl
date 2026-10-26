---
description: simulatePlayers domain stage for Escape Room Foundry.
args:
  goal: string
  records: PuzzleArtifact[]
  evidence: EvidenceItem[]
returns: StageResultPuzzlePlaytest
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPuzzlePlaytest values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Run exact state-space checks and simulated player interpretations, including out-of-order discovery, lost props, repeated actions, and plausible wrong answers. Identify soft locks and confusing clues, then propose targeted repairs rather than adding arbitrary hints everywhere. The stage gate is: Every reachable operational state has a safe exit and a solvable or staff-recoverable route.
