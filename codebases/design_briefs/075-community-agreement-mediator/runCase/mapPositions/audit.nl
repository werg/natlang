---
description: Independently audit a mapPositions draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftMediationPosition }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Each position is confirmed or clearly marked as an unconfirmed interpretation.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
