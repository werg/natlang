---
description: Advance knowledge only through witnessed scenes and source-authorized message edges.
args: { graph: SceneRevealGraph; showEvidence: EvidenceItem[] }
returns: KnowledgeLedger
---
1. Start separate byCharacter and byAudienceGroup maps, protectedFacts and unprovenTransfers. Read the source-established initial knowledge and actual witnessed scene disclosures for the reviewed production. A current script establishes what a scene may reveal; it does not establish that any participant has already witnessed it.
2. Walk the graph along each evidenced audience route and character history. For each edge, match its source IDs, scene identities, permitted recipients and approval revision. Add its fact only to a named recipient whose actual attendance, message receipt or established prior knowledge is evidenced. Future, unexecuted and disputed edges remain potential reveals and do not enter the current knowledge maps.
3. Preserve a fact as protected when the source establishes its restriction. Record the precise edge and missing observation in unprovenTransfers when a required present knowledge claim cannot be established. Lack of an optional future receipt is not a present knowledge contradiction.
4. Return KnowledgeLedger { byCharacter, byAudienceGroup, protectedFacts, unprovenTransfers }. Repeated observations of the same fact add no duplicate entry. Never union audience groups, assign an unknown recipient globally, or let a quiet-route attendee inherit a speaking-role disclosure.
