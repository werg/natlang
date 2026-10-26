---
description: Independently audit a modelDramaturgy draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTheatreScene }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Scene transitions respect knowledge and narrative prerequisites.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
