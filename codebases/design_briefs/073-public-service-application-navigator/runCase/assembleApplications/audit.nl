---
description: Independently audit a assembleApplications draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftApplicationTask }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Application content is evidence-backed and explicitly confirmed by the applicant where required.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
