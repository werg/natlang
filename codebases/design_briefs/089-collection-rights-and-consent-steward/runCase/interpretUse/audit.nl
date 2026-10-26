---
description: Independently audit a interpretUse draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftProposedCollectionUse }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Proposed use is specific enough to compare with the governing records.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
