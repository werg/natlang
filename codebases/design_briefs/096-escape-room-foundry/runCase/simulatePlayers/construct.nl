---
description: Construct a PuzzlePlaytest draft for simulatePlayers.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PuzzleArtifact[] }
returns: DraftPuzzlePlaytest
---
Assemble the PuzzlePlaytest as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every reachable operational state has a safe exit and a solvable or staff-recoverable route.
