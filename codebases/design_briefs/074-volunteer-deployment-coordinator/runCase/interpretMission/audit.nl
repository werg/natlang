---
description: Independently audit a interpretMission draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftMissionRequirement }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Requirements are task-based and not inferred from volunteer demographics.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
