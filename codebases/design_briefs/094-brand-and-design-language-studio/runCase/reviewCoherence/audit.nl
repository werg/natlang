---
description: Independently audit a reviewCoherence draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftBrandReview }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No unsupported claim or unresolved hard inconsistency; originality receives separate human evaluation.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
