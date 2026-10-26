---
description: Independently audit a packageOperations draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftEscapeOperations }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Independent solvability, safe operation, reliable reset, and human enjoyment review all required.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
