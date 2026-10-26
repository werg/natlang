---
description: Independently audit a judgeIdentity draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftHistoricalIdentityFinding }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Established links require discriminating evidence and compatible chronology.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
