---
description: Construct a PreparationContext draft for interpretCircumstances.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: PreparationRequirement[] }
returns: DraftPreparationContext
---
Assemble the PreparationContext as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Context uses volunteered facts and does not change the clinical plan.
