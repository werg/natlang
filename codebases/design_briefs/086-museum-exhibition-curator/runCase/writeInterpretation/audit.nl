---
description: Independently audit a writeInterpretation draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftExhibitionLabel }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every material claim is supported or explicitly framed as interpretation.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
