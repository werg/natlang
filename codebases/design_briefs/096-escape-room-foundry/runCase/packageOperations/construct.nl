---
description: Construct a EscapeOperations draft for packageOperations.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PuzzlePlaytest[] }
returns: DraftEscapeOperations
---
Assemble the EscapeOperations as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Independent solvability, safe operation, reliable reset, and human enjoyment review all required.
