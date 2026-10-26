---
description: Independently audit a alignConcepts draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftMultilingualConcept }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Concept alignment is supported by definitions and counterexamples, not dictionary matching alone.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
