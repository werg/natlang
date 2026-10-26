---
description: Independently audit a linkJourneyEvents draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftJourneyLink }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Patient isolation and evidence-backed episode linkage.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
