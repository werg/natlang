---
description: Follow contextual citations without inferring the contents of absent records or exceeding effort.
args: { request:CommissionRequest; sources:ArchiveSource[]; observations:SearchObservation[]; checkpoint:ResearchCheckpoint }
returns: FollowupLead[]
---
Bind lineages = `compareCitationAncestry(request, sources, observations)` before ranking leads. For each hit, connect supplied linked identifiers to the question they might answer; use the lineages' root identifiers to avoid counting derivative accounts as independent witnesses. Missing identifiers become missing-source leads. Rank leads by relevance and effort. Preserve an unaffordable lead as open and request an extension instead of silently spending beyond the commission. Return the exact lineages' unresolved source IDs on their corresponding FollowupLead rationale.
