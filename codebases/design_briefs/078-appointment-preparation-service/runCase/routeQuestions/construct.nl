---
description: Construct a PreparationQuestion draft for routeQuestions.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PreparationChecklist[] }
returns: DraftPreparationQuestion
---
Assemble the PreparationQuestion as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Clinical clarification comes from an authorized source.
