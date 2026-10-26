---
description: Construct a CreativeDirection draft for generateDirections.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: BrandConstraint[] }
returns: DraftCreativeDirection
---
Assemble the CreativeDirection as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Directions satisfy factual constraints and are meaningfully distinct.
