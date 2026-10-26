---
description: Construct a KitchenProductionPlan draft for planAndCost.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MenuComposition[] }
returns: DraftKitchenProductionPlan
---
Assemble the KitchenProductionPlan as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact quantities, costs, equipment capacity, and approved handling constraints reconcile.
