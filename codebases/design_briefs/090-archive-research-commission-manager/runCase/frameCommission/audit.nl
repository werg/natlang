---
description: Independently audit a frameCommission draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftResearchCommissionScope }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Each subquestion has a plausible evidence path and an explicit inferential limit.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
