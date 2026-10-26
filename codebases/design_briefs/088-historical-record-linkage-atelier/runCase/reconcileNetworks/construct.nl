---
description: Construct a HistoricalNetwork draft for reconcileNetworks.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: HistoricalIdentityFinding[] }
returns: DraftHistoricalNetwork
---
Assemble the HistoricalNetwork as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No self-supporting evidence cycle or established chronological contradiction.
