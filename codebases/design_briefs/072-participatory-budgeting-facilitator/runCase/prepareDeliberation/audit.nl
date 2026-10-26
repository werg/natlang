---
description: Independently audit a prepareDeliberation draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftBudgetDecisionPacket }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Selection follows the published process and the selected portfolio matches its approved scope.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
