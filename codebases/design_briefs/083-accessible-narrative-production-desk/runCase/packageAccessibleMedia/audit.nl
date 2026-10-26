---
description: Independently audit a packageAccessibleMedia draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftAccessibilityPackage }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (All deliverables align to the same approved media timeline and version.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
