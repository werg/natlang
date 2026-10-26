---
description: Construct a CounterpartAction draft for runCounterparts.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TrainingOffer[] }
returns: DraftCounterpartAction
---
Assemble the CounterpartAction as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Role knowledge, mandate, and conduct constraints hold throughout the episode.
