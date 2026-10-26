---
description: Independently audit a rewriteStructure draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftPlainLanguageDraft }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Rewrite preserves every material condition, exception, and uncertainty.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
