---
description: Construct a PuzzleNode draft for constructPuzzleGraph.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EscapeConstraint[] }
returns: DraftPuzzleNode
---
Assemble the PuzzleNode as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact reachability and acyclic prerequisite checks; all mandatory goals have a feasible path.
