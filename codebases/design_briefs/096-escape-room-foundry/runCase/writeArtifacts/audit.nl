---
description: Independently audit a writeArtifacts draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftPuzzleArtifact }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Clues are sufficient, internally consistent, and do not require inaccessible external knowledge.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
