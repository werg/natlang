---
description: Independently audit a verifyEdition draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftStoryEdition }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Global continuity and approved narrative constraints pass; creative quality receives separate editorial review.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
