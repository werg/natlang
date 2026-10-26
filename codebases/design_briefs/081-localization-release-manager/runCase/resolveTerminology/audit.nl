---
description: Independently audit a resolveTerminology draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftLocaleTermDecision }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Term choice is concept- and context-specific, with versioned authority.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
