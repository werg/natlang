---
description: Independently audit a qualifyIssue draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftIssueReadiness }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Complete issue meets scope, rights, factual, and production gates.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
