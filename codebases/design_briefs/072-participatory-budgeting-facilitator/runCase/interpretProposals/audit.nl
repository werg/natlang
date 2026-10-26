---
description: Independently audit a interpretProposals draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftCivicProposal }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Proposal normalization does not silently change the project’s purpose.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
