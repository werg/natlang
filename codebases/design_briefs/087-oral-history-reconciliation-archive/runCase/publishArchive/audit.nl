---
description: Independently audit a publishArchive draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftOralHistoryEdition }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No restricted content leaks through summaries, search snippets, or merged narratives.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
