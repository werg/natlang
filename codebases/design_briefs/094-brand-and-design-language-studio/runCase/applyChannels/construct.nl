---
description: Construct a BrandAsset draft for applyChannels.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: BrandSystem[] }
returns: DraftBrandAsset
---
Assemble the BrandAsset as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Each asset is useful in its channel and consistent with the approved system.
