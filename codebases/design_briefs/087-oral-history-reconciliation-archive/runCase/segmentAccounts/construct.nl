---
description: Construct a OralHistoryAccount draft for segmentAccounts.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: OralHistoryProject[] }
returns: DraftOralHistoryAccount
---
Assemble the OralHistoryAccount as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Attribution and evidentiary character remain explicit.
