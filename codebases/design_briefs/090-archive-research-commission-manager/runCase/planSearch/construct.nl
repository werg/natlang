---
description: Construct a ArchiveSearchPlan draft for planSearch.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ResearchCommissionScope[] }
returns: DraftArchiveSearchPlan
---
Assemble the ArchiveSearchPlan as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Search plan distinguishes accessible, inaccessible, unsearched, and exhausted collections.
