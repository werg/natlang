---
description: Construct a DocumentaryClearance draft for reviewProduction.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: DocumentaryEdit[] }
returns: DraftDocumentaryClearance
---
Assemble the DocumentaryClearance as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Required factual and rights gates pass; open issues remain visible.
