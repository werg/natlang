---
description: Independently audit a coordinateLiveChange draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftEventChange }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Safety authority, staff acknowledgment, and public communications align with the current plan.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
