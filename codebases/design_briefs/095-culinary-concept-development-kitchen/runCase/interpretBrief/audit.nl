---
description: Independently audit a interpretBrief draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftMenuRequirement }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Mandatory dietary and operational constraints are explicit and source-backed.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
