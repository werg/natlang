---
description: Follow contextual citations without inferring the contents of absent records or exceeding effort.
args: { request:CommissionRequest; sources:ArchiveSource[]; observations:SearchObservation[]; checkpoint:ResearchCheckpoint }
returns: FollowupLead[]
---
For each hit, connect supplied linked identifiers to the question they might answer. Missing identifiers become missing-source leads. Compare ancestry: if two accounts only cite one absent source, retain one evidential lineage. Rank leads by relevance and effort. Preserve an unaffordable lead as open and request an extension instead of silently spending beyond the commission.
