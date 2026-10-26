---
description: Independently audit a interpretCollection draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCuratorialEvidence }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Every proposed fact and display use has evidence and applicable permission.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
