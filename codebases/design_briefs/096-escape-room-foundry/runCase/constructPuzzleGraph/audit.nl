---
description: Independently audit a constructPuzzleGraph draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftPuzzleNode }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Exact reachability and acyclic prerequisite checks; all mandatory goals have a feasible path.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
