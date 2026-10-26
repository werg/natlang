---
description: Independently audit a constructSequence draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTransitionTask }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (All logistical prerequisites are feasible and traceable to the approved plan.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
