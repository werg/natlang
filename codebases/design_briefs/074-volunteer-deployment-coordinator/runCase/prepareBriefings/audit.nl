---
description: Independently audit a prepareBriefings draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftVolunteerBriefing }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Briefings preserve approved procedures and contain no unauthorized personal disclosures.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
