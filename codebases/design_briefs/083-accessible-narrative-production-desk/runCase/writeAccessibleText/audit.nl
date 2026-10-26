---
description: Independently audit a writeAccessibleText draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftAccessibleSegment }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Text is accurate, appropriately timed, and does not spoil future information.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
