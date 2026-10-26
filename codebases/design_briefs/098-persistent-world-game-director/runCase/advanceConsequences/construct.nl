---
description: Construct a WorldConsequence draft for advanceConsequences.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CharacterAction[] }
returns: DraftWorldConsequence
---
Assemble the WorldConsequence as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Resource conservation, chronology, and knowledge propagation remain valid.
