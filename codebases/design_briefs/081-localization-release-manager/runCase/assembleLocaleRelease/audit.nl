---
description: Independently audit a assembleLocaleRelease draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftLocaleReleasePacket }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Bundle is internally version-consistent and every fallback is declared and safe.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
