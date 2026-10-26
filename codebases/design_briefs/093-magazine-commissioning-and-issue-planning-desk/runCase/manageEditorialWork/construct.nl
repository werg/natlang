---
description: Construct a EditorialTask draft for manageEditorialWork.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: CommissionBrief[] }
returns: DraftEditorialTask
---
Assemble the EditorialTask as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Required reporting and evidence obligations are fulfilled, not merely polished prose.
