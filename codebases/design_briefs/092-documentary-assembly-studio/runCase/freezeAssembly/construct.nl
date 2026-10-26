---
description: Construct a DocumentaryRelease draft for freezeAssembly.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: DocumentaryClearance[] }
returns: DraftDocumentaryRelease
---
Assemble the DocumentaryRelease as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: All deliverables refer to the same approved assembly and licensed source versions.
