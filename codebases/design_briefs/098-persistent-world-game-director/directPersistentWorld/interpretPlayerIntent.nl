---
description: Map utterance into bounded action candidates without treating narration as executed state.
args:{world:PersistentWorldState;utterance:PlayerUtterance;sourceRecord:GameWorld;sourceEvidence:EvidenceItem[]}
returns:{actions:LegalAction[];questions:string[];unresolved:string[];}
---
Bind worldId, ruleSetId, sessionIds, directorId and contentBoundaryIds from the top-level GameWorld record. Check the active world and rule version against those identities before parsing the request. Apply the current content-boundary identifiers to the candidate actions and never regress to the fixture’s generic world or boundary labels.

Use visible location and world context available to the player. Separate intention, speech and requested outcome. Generate only action types allowed by the current rules. Ask a question when ambiguity changes target, resource commitment or irreversible consequence. Do not grant a capability because the player asserts success.
