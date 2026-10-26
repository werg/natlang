---
description: Construct a HistoricalIdentityFinding draft for judgeIdentity.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: HistoricalCandidate[] }
returns: DraftHistoricalIdentityFinding
---
Assemble the HistoricalIdentityFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Established links require discriminating evidence and compatible chronology.
