---
description: Construct a BrandReview draft for reviewCoherence.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: BrandAsset[] }
returns: DraftBrandReview
---
Assemble the BrandReview as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: No unsupported claim or unresolved hard inconsistency; originality receives separate human evaluation.
