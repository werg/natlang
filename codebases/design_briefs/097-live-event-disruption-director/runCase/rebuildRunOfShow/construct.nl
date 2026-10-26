---
description: Construct a EventRunPlan draft for rebuildRunOfShow.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EventRecoveryOption[] }
returns: DraftEventRunPlan
---
Assemble the EventRunPlan as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No double booking, impossible setup sequence, or unsupported timing promise.
