---
description: Independently audit a constructPortfolios draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCivicPortfolio }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Exact budget feasibility and complete disclosure of exclusions and dependencies.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
