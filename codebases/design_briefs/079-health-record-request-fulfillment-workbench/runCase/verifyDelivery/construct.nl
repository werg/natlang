---
description: Construct a RecordDelivery draft for verifyDelivery.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: RecordFulfillment[] }
returns: DraftRecordDelivery
---
Assemble the RecordDelivery as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Correct recipient, current authority, and exact released artifact identity are verified.
