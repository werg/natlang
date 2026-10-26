---
description: Independently audit a negotiateDecision draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTerminologyDecision }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Decision authority and applicability are explicit; unresolved conceptual conflict remains visible.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
