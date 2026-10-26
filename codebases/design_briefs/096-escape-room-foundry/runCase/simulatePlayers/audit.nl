---
description: Independently audit a simulatePlayers draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftPuzzlePlaytest }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every reachable operational state has a safe exit and a solvable or staff-recoverable route.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
