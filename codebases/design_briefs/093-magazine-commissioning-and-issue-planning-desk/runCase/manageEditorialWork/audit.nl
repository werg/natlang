---
description: Independently audit a manageEditorialWork draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftEditorialTask }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Required reporting and evidence obligations are fulfilled, not merely polished prose.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
