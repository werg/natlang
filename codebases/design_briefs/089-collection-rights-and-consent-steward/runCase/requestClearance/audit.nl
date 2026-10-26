---
description: Independently audit a requestClearance draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftClearanceTask }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Clearance closes a gap only with appropriate documented authority.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
