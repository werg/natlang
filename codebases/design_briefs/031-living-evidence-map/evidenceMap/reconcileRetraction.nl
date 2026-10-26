---
args: { claims: EvidenceClaimGraph, retractions: EvidencePage[] }
returns: ReviewUpdateLedger
---
Bind a retraction or correction to canonical study lineage and determine which claims depend on the affected report. Invalidate only dependent claims, preserve unaffected claims and prior review history, and identify review owners who need notice.
