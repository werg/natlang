---
description: Independently audit a qualifyExhibition draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftExhibitionRelease }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (All required specialist approvals and exact physical constraints pass before installation.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
