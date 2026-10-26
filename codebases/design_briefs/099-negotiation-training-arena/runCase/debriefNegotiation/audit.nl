---
description: Independently audit a debriefNegotiation draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftNegotiationDebrief }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Feedback cites actual behavior and does not reward coercion or unsupported inference about motives.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
