---
description: Construct a LocalizedSegment draft for transformContent.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: LocaleTermDecision[] }
returns: DraftLocalizedSegment
---
Assemble the LocalizedSegment as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Meaning, protected tokens, and required product terminology survive transformation.
