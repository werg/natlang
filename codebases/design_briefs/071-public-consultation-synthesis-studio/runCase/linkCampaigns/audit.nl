---
description: Independently audit a linkCampaigns draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftSubmissionRelation }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Template use does not invalidate a participant’s submission or erase unique content.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
