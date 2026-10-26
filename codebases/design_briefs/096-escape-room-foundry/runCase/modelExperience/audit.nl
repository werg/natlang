---
description: Independently audit a modelExperience draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftEscapeConstraint }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No real confinement, unsafe mechanism, or inaccessible mandatory path under supplied requirements.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
