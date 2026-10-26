---
description: Construct a NegotiationScenario draft for authorNegotiation.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: NegotiationExercise[] }
returns: DraftNegotiationScenario
---
Assemble the NegotiationScenario as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Exact feasibility and learning-objective coverage verified before use.
