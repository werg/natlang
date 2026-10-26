---
description: Construct a HistoricalObservation draft for normalizeRecords.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: HistoricalLinkage[] }
returns: DraftHistoricalObservation
---
Assemble the HistoricalObservation as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Original representation and uncertainty remain available.
