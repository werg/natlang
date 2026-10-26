---
description: Independently audit a requestMaterial draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftReferralRequest }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Requests are minimal, authorized, and tied to a specific requirement.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
