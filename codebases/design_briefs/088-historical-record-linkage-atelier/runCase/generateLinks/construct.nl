---
description: Construct a HistoricalCandidate draft for generateLinks.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: HistoricalObservation[] }
returns: DraftHistoricalCandidate
---
Assemble the HistoricalCandidate as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Candidate generation is bounded and recall is independently assessed.
