---
description: Construct a MenuRequirement draft for interpretBrief.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MenuProject[] }
returns: DraftMenuRequirement
---
Assemble the MenuRequirement as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Mandatory dietary and operational constraints are explicit and source-backed.
