---
description: Construct a OralHistoryReference draft for resolveReferences.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: OralHistoryAccount[] }
returns: DraftOralHistoryReference
---
Assemble the OralHistoryReference as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Identity links are evidence-backed or remain provisional.
