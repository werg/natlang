---
description: Independently audit a verifyDelivery draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftRecordDelivery }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Correct recipient, current authority, and exact released artifact identity are verified.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
