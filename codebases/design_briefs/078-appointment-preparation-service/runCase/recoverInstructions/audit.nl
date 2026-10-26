---
description: Independently audit a recoverInstructions draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftPreparationRequirement }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every consequential instruction and exception remains traceable to the approved source.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
