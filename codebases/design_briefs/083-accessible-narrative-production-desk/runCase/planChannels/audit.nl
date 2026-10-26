---
description: Independently audit a planChannels draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftAccessibilityChannelPlan }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Channel plan preserves essential information within actual timing and capacity constraints.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
