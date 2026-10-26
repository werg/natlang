---
description: Independently audit a mapCoverage draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftIssueCoverage }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Coverage choices follow the issue brief and available evidence.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
