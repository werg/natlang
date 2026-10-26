---
description: Independently audit a reviewProduction draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftDocumentaryClearance }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Required factual and rights gates pass; open issues remain visible.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
