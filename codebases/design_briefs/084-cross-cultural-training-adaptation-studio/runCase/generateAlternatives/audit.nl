---
description: Independently audit a generateAlternatives draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftAdaptedScenario }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Instructional equivalence and contextual plausibility are separately reviewed.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
