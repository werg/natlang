---
description: Construct a NegotiationDebrief draft for debriefNegotiation.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CounterpartAction[] }
returns: DraftNegotiationDebrief
---
Assemble the NegotiationDebrief as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Feedback cites actual behavior and does not reward coercion or unsupported inference about motives.
