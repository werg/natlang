---
description: Construct a DocumentaryStructure draft for constructArguments.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: DocumentarySegment[] }
returns: DraftDocumentaryStructure
---
Assemble the DocumentaryStructure as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Narrative argument remains supportable by the selected material.
