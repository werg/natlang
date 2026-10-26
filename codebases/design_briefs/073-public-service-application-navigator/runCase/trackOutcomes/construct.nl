---
description: Construct a ServiceApplicationStatus draft for trackOutcomes.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ApplicationTask[] }
returns: DraftServiceApplicationStatus
---
Assemble the ServiceApplicationStatus as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No claim of benefit approval without authoritative confirmation.
