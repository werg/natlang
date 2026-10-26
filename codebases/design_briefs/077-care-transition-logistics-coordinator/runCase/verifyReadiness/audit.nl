---
description: Independently audit a verifyReadiness draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTransitionReadiness }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No transition-ready claim while a required logistical prerequisite is unverified.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
