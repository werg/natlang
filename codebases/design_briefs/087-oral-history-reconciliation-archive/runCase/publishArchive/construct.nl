---
description: Construct a OralHistoryEdition draft for publishArchive.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: OralHistoryQuestion[] }
returns: DraftOralHistoryEdition
---
Assemble the OralHistoryEdition as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No restricted content leaks through summaries, search snippets, or merged narratives.
