---
description: Independently audit a indexMaterial draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftDocumentarySegment }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every segment has source timecodes, context, and permitted-use status.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
