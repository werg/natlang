---
description: Independently audit a reviewTastings draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCulinaryRevision }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Revised menu passes hard checks; culinary quality remains chef-reviewed rather than self-certified.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
