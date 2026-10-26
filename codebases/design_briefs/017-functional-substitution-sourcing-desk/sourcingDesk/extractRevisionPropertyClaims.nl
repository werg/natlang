---
args: { candidates: StageResultSubstituteCandidate, pages: EvidencePage[] }
returns: CandidatePropertyEvidence[]
---
Extract a property claim only when the page explicitly identifies candidate item, supplier, revision, property name, value, and unit. Bind every claim to its source Ref. Do not transfer a prior or similar catalog item's claim to the current candidate.
