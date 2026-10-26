---
description: Construct a DishConcept draft for generateDishes.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MenuRequirement[] }
returns: DraftDishConcept
---
Assemble the DishConcept as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Concepts respect verified restrictions and documented kitchen capability.
