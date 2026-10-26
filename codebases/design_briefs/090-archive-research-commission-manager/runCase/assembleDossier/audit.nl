---
description: Independently audit a assembleDossier draft.
args: { goal: string; slice: EvidenceSlice; draft: DraftResearchDossier }
returns: StageAudit
---
Call qualify with the evidence slice and draft. Then conduct an independent counterinterpretation review: check source attribution, contradictions, freshness, scope, and the stage gate (Another researcher can reproduce the searched scope and distinguish evidence from inference.). If qualify fails, retain its precise finding. If evidence is insufficient or materially conflicting, return needs-review. Return smallest repair targets; never silently drop a candidate.
