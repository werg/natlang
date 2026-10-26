---
description: Construct a ClearanceTask draft for requestClearance.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: RightsFinding[] }
returns: DraftClearanceTask
---
Assemble the ClearanceTask as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Clearance closes a gap only with appropriate documented authority.
