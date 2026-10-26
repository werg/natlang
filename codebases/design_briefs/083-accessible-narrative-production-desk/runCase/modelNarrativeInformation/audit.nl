---
description: Independently audit a modelNarrativeInformation draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftNarrativeInformation }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (No information is invented beyond supplied media evidence or revealed prematurely.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
