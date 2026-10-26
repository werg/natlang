---
description: Independently audit a releaseAccessible draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftPlainLanguageRelease }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Independent fidelity review passes; readability alone is not the acceptance criterion.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
