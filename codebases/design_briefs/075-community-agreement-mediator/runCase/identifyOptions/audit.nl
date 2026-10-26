---
description: Independently audit a identifyOptions draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCommunityOption }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Options are feasible and accurately represent who would give up what.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
