---
description: Independently audit a propagateTerminology draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftTerminologyImpact }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No global search-and-replace across different senses; affected content passes fidelity checks.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
