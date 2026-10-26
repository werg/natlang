---
description: Independently audit a reviewContent draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftRecordDisclosureFinding }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every exclusion or redaction has a supplied policy basis and review status.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
