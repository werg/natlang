---
description: Construct a CollectionUseRecord draft for maintainUseRegister.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ClearanceTask[] }
returns: DraftCollectionUseRecord
---
Assemble the CollectionUseRecord as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Approved use, actual distribution, attribution, and duration remain aligned.
