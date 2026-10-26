---
description: Construct a ServiceMechanism draft for formMechanisms.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: JourneyLink[] }
returns: DraftServiceMechanism
---
Assemble the ServiceMechanism as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Mechanisms remain appropriately qualified and source-supported.
