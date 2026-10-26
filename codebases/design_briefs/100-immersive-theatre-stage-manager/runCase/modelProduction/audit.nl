---
description: Independently audit a modelProduction draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTheatreConstraint }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Required resources and consent conditions are explicit and current.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
