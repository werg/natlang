---
description: Independently audit a releaseTraining draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftAdaptedTrainingRelease }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Approved audience scope and preserved learning objectives are explicit.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
