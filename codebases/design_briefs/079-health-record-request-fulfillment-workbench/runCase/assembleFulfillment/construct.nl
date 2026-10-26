---
description: Construct a RecordFulfillment draft for assembleFulfillment.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: RecordDisclosureFinding[] }
returns: DraftRecordFulfillment
---
Assemble the RecordFulfillment as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No recoverable restricted content, missing unexplained item, or mismatched patient document.
