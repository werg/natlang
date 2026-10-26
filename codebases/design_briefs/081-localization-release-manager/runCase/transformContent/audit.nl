---
description: Independently audit a transformContent draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftLocalizedSegment }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Meaning, protected tokens, and required product terminology survive transformation.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
