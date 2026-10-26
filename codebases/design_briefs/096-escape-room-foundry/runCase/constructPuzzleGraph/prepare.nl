---
description: Prepare one constructPuzzleGraph evidence slice.
args: { goal: string; records: EscapeConstraint[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Design a dependency graph of puzzles and information reveals, combining parallel and sequential paths deliberately. Give each node a meaningful narrative purpose, avoid circular prerequisites, and ensure every required clue is discoverable before it is needed. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
