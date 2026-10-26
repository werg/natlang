---
description: Independently audit a runCounterparts draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCounterpartAction }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Role knowledge, mandate, and conduct constraints hold throughout the episode.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
