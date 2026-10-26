---
description: Independently audit a assessCompatibility draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftRightsFinding }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every allowed use has positive authority and all applicable conditions are satisfied.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
