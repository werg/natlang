---
description: Independently audit a recoverBrandConstraints draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftBrandConstraint }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Product claims and prohibitions have authoritative sources.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
