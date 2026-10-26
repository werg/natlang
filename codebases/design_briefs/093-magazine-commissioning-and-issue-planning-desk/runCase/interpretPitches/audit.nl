---
description: Independently audit a interpretPitches draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftEditorialPitch }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Pitch evaluation is grounded in the supplied editorial goals and feasible reporting.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
