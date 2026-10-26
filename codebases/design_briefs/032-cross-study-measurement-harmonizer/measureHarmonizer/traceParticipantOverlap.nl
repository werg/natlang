---
args: { request: HarmonizationRequest, evidence: EvidencePage[] }
returns: string[]
---
Trace participant overlap only through authorized linkage evidence. Preserve uncertainty and privacy boundary for unresolved overlap; do not count shared participants as independent observations or create identity joins from demographic similarity.
