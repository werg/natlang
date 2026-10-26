---
description: Independently audit a modelReaderTasks draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftReaderTask }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Reader tasks cover all consequential uses of the document.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
