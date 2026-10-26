---
description: Independently audit a interpretAuthorization draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftRecordAuthorization }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Current authorization covers the exact recipient and proposed disclosure purpose.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
