---
description: Construct a LocalizationSegment draft for segmentWithContext.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: LocalizationRelease[] }
returns: DraftLocalizationSegment
---
Assemble the LocalizationSegment as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact token and markup structure preserved; context is not discarded during deduplication.
