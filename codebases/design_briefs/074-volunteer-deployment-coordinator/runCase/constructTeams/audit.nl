---
description: Independently audit a constructTeams draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftVolunteerTeam }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (All required capabilities and consent conditions satisfied or the mission remains blocked.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
