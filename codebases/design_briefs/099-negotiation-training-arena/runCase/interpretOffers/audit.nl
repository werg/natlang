---
description: Independently audit a interpretOffers draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTrainingOffer }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Exact terms and authority limits are preserved before any simulated commitment.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
