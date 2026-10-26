---
description: Preserve a negotiation checkpoint unchanged when a stale or divergent resume request cannot be reconciled.
args:{scenario:ArenaScenario;checkpoint:ArenaCheckpoint;turns:ArenaTurn[];feasibility:FeasibilityArtifact[];questions:string[]}
returns:ArenaRunResult
---
Return the supplied checkpoint without changing its revision, turns, openOfferIds, acceptedTerms, difficultyRevision or debriefReleaseScope. Return the supplied caller history only as visible context if needed, and include every lineage question as a clarification question. Preserve feasibility artifacts as supplied without claiming one applies to the current scenario. Set private feedback and debrief to empty because no new authorized turn was processed, and mark status clarification-needed. This is a hold result; do not parse a new offer, choose a counterpart action or alter the durable episode.
