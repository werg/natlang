---
description: Independently audit a recoverMeaning draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftMeaningUnit }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (All operative meaning units are represented with source locations.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
