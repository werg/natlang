---
description: Construct a PreparationChecklist draft for composeChecklist.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PreparationContext[] }
returns: DraftPreparationChecklist
---
Assemble the PreparationChecklist as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Semantic fidelity to approved instructions passes independent review.
