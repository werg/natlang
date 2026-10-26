---
description: constructPuzzleGraph domain stage for Escape Room Foundry.
args:
  goal: string
  records: EscapeConstraint[]
  evidence: EvidenceItem[]
returns: StageResultPuzzleNode
---
For each record candidate, call prepare({ goal: goal, records: [candidate], evidence: evidence }) to create one bounded slice. Call decide({ goal: goal, slice: preparedSlice }); call construct({ goal: goal, slice: preparedSlice, decisions: decisionAtoms, prior: records }); then call audit({ goal: goal, slice: preparedSlice, draft: constructedRecord }). The audit invokes the finite qualify gate and an independent counterinterpretation check. Place only passing records in accepted; put unresolved or needs-review DraftPuzzleNode values in unresolved with exact missingFields, conflicts, and evidenceIds; do not invent absent values; put affirmatively contradicted candidates in rejected with the reason. Account for every frozen candidate in the StageResult. Use only this stage's method: Design a dependency graph of puzzles and information reveals, combining parallel and sequential paths deliberately. Give each node a meaningful narrative purpose, avoid circular prerequisites, and ensure every required clue is discoverable before it is needed. The stage gate is: Exact reachability and acyclic prerequisite checks; all mandatory goals have a feasible path.
