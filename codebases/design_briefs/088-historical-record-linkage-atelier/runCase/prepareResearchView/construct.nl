---
description: Construct a HistoricalLinkageEdition draft for prepareResearchView.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: HistoricalNetwork[] }
returns: DraftHistoricalLinkageEdition
---
Assemble the HistoricalLinkageEdition as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Accepted and provisional relationships remain distinguishable in every export.
