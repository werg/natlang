---
description: Independently audit a replayLearning draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftAdaptationTest }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (The same competency distinctions remain observable in both versions.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
