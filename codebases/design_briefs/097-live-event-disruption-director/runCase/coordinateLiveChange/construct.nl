---
description: Construct a EventChange draft for coordinateLiveChange.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: EventRunPlan[] }
returns: DraftEventChange
---
Assemble the EventChange as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Safety authority, staff acknowledgment, and public communications align with the current plan.
