---
description: Independently audit a replanAssignments draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftVolunteerRevision }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Replanned teams retain required coverage, supervision, and current consent.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
