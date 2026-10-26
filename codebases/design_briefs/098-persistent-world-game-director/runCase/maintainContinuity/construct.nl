---
description: Construct a WorldContinuity draft for maintainContinuity.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: WorldConsequence[] }
returns: DraftWorldContinuity
---
Assemble the WorldContinuity as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No retroactive state invention or unapproved content-boundary change.
