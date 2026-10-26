---
description: Independently audit a verifySynthesis draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftConsultationReport }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No material theme omission, misleading denominator, or privacy leak in published evidence.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
