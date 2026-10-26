---
description: Independently audit a formResponses draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftConsultationResponseOption }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Response accurately represents the concern and does not claim unsupported public consensus.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
