---
description: Independently audit a reviewResults draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftExperienceOutcome }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Reported improvements match exact measures and preserve limits and counterevidence.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
