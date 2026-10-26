---
description: Independently audit a recoverLearningMechanism draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTrainingMechanism }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every scenario has an explicit instructional mechanism before adaptation.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
