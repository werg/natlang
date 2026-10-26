---
description: Independently audit a extractLogistics draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTransitionRequirement }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No generated change to clinical content or priority.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
