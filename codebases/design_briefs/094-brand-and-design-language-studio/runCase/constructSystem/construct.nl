---
description: Construct a BrandSystem draft for constructSystem.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CreativeDirection[] }
returns: DraftBrandSystem
---
Assemble the BrandSystem as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Rules are operationally usable and examples do not contradict them.
