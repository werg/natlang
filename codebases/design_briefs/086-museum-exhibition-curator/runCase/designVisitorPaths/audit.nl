---
description: Independently audit a designVisitorPaths draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftExhibitionZone }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Physical layout and artifact constraints are feasible under supplied venue data.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
