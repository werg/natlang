---
description: Construct a PuzzleArtifact draft for writeArtifacts.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PuzzleNode[] }
returns: DraftPuzzleArtifact
---
Assemble the PuzzleArtifact as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Clues are sufficient, internally consistent, and do not require inaccessible external knowledge.
