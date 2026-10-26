---
description: Construct a RecordAuthorization draft for interpretAuthorization.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: HealthRecordRequest[] }
returns: DraftRecordAuthorization
---
Assemble the RecordAuthorization as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Current authorization covers the exact recipient and proposed disclosure purpose.
