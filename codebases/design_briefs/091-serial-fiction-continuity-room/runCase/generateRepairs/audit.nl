---
description: Independently audit a generateRepairs draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftStoryRevision }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Repairs satisfy hard canon constraints unless a reviewed canon change is explicitly proposed.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
