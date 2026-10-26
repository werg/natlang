---
args: { sample: SamplingManifest, pilot: EvidencePage[] }
returns: CodebookConstruction
---
Derive candidate codes only from passages in the frozen pilot sample. Keep each proposed code as a separate candidate with observable description, inferred mechanism, and cited passage examples. For each candidate call reviewCodeBoundary(candidate, sample, pilot). Use the child's actual boundary and disposition; do not pass an unresolved boundary to coding as if it were stable. Return { boundaries: the reviewed boundaries including unresolved ones with their exact versions, reviews: the per-code boundary reviews }. Preserve overlapping meanings when collapsing them would erase distinct participant experiences.
