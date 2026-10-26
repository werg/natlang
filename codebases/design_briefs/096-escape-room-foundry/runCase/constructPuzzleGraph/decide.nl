---
description: Decide field-level findings for constructPuzzleGraph.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Design a dependency graph of puzzles and information reveals, combining parallel and sequential paths deliberately. Give each node a meaningful narrative purpose, avoid circular prerequisites, and ensure every required clue is discoverable before it is needed. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
