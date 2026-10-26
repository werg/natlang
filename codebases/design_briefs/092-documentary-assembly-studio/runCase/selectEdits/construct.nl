---
description: Construct a DocumentaryEdit draft for selectEdits.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: DocumentaryStructure[] }
returns: DraftDocumentaryEdit
---
Assemble the DocumentaryEdit as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No materially misleading quote edit or unsupported causal implication.
