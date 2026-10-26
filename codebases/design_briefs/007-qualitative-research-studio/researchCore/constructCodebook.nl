---
args: { sample: SamplingManifest, pilot: EvidencePage[] }
returns: CodeBoundary[]
---
Derive candidate codes only from passages in the frozen pilot sample. For each code, separate observable description from explanatory interpretation, bind inclusion and exclusion examples to passage references, and search the pilot for a boundary counterexample. Preserve overlapping meanings when collapsing them would erase distinct participant experiences; do not claim a stable code before examples establish its boundary.
