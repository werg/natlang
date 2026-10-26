---
description: Enumerate scene route alternatives and their distinct resource and disclosure costs.
args: { incident: ShowIncident; dramaturgy: DramaturgyMap; resources: ResourceTimeline; showEvidence: EvidenceItem[] }
returns: VariantCandidate[]
---
Find each rehearsed alternative for the affected scenes in current approved evidence. For every route, return its stable variant ID, exact preserved dramatic beat, cast and prop requirements, fact disclosures by audience group, participant actions, and rehearsal sign-off evidence. Label an unapproved rewrite as `rewrite-review-required`; never rank it above a signed-off variant just because its prose is better. If no route data exists, return no candidate and leave the scene blocked.
