---
description: Construct a CharacterView draft for modelCharacterViews.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PlayerIntent[] }
returns: DraftCharacterView
---
Assemble the CharacterView as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Character-visible context contains only authorized knowledge.
