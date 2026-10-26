---
description: Independently audit a assembleFulfillment draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftRecordFulfillment }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No recoverable restricted content, missing unexplained item, or mismatched patient document.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
