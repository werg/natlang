---
description: Independently audit a generateLinks draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftHistoricalCandidate }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Candidate generation is bounded and recall is independently assessed.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
