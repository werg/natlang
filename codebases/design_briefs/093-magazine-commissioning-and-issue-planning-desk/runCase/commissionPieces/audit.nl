---
description: Independently audit a commissionPieces draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCommissionBrief }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Contributor accepts the exact scope and commercial conditions before work is committed.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
