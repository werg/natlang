---
description: Construct a CulinaryRevision draft for reviewTastings.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: KitchenProductionPlan[] }
returns: DraftCulinaryRevision
---
Assemble the CulinaryRevision as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Revised menu passes hard checks; culinary quality remains chef-reviewed rather than self-certified.
