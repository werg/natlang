---
description: Find continuity conflicts between new scene and temporal canon state.
args: { edition:SerialEdition; proposal:SerialScene; canon:CanonAssertion[]; knowledge:KnowledgeEvent[]; obligations:ArcObligation[] }
returns: ContinuityFinding[]
---
Compare the proposed scene's position with the episode chronology and each character's knowledge at that exact boundary. Check only applicable world facts and motivations. Compare open obligations with the scene's consequences; distinguish an intentionally open mystery from a dropped setup using its approved rationale. Classify a reinterpreting reveal separately from a contradiction and cite every source scene that creates the conflict.
