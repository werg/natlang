---
description: Independently audit a applyChannels draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftBrandAsset }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Each asset is useful in its channel and consistent with the approved system.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
