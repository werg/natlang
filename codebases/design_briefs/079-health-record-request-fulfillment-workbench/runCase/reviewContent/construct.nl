---
description: Construct a RecordDisclosureFinding draft for reviewContent.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: RecordSelection[] }
returns: DraftRecordDisclosureFinding
---
Assemble the RecordDisclosureFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Every exclusion or redaction has a supplied policy basis and review status.
