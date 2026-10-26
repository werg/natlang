---
description: Independently audit a qualifyFeasibility draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftProposalFeasibility }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Eligibility and feasibility decisions cite supplied rules and verified estimates.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
