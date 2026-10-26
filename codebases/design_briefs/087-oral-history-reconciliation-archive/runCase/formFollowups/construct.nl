---
description: Construct a OralHistoryQuestion draft for formFollowups.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: RecollectionComparison[] }
returns: DraftOralHistoryQuestion
---
Assemble the OralHistoryQuestion as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Questions are nonleading, relevant, and within consent scope.
