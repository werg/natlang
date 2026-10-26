---
description: Construct a EventRecoveryOption draft for generateFormats.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EventDisruption[] }
returns: DraftEventRecoveryOption
---
Assemble the EventRecoveryOption as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Alternatives preserve hard safety and accessibility constraints.
