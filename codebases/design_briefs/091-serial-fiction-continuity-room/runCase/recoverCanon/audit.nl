---
description: Independently audit a recoverCanon draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCanonFact }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every canon fact has source and status, including belief versus world fact.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
