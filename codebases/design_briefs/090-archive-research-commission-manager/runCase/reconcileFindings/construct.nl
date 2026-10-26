---
description: Construct a ArchiveFinding draft for reconcileFindings.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ArchiveLead[] }
returns: DraftArchiveFinding
---
Assemble the ArchiveFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Conclusions remain within actual evidence and search coverage.
