---
description: Independently audit a chooseWorldActions draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCharacterAction }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Action is legal and dialogue does not reveal unknown or restricted facts.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
