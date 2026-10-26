---
description: Independently audit a interpretRequirements draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftReferralRequirement }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Requirements are current, service-specific, and administrative in scope.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
