---
description: Construct a MenuComposition draft for constructMenu.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: DishConcept[] }
returns: DraftMenuComposition
---
Assemble the MenuComposition as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Menu-level feasibility and required dietary coverage pass.
