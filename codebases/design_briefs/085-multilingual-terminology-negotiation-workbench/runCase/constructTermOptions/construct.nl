---
description: Construct a TermOption draft for constructTermOptions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: MultilingualConcept[] }
returns: DraftTermOption
---
Assemble the TermOption as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Each option preserves the intended concept in its stated context.
