---
description: Independently audit a segmentWithContext draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftLocalizationSegment }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Exact token and markup structure preserved; context is not discarded during deduplication.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
