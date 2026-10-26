---
description: Independently audit a prepareRoles draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftNegotiationRole }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No cross-role leakage of private preferences or hidden scoring data.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
