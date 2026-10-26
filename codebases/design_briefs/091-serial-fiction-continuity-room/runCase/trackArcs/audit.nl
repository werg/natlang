---
description: Independently audit a trackArcs draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftStoryObligation }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Arc claims are grounded in the actual text or approved author intent.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
