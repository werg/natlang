---
description: Construct a NegotiationRole draft for prepareRoles.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: NegotiationScenario[] }
returns: DraftNegotiationRole
---
Assemble the NegotiationRole as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No cross-role leakage of private preferences or hidden scoring data.
