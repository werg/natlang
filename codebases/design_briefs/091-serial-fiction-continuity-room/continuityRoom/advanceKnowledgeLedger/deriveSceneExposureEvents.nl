---
description: Derive character knowledge events from approved scene order and explicit exposure evidence.
args:{edition:SerialEdition;proposal:SerialScene}
returns:{events:KnowledgeEvent[];invalidatedEventIds:string[];unresolvedExposureIds:string[]}
---
Walk approved scenes in episode/order sequence and retain the scene IDs that explicitly reveal each secret. For each character, create an event only when the scene's characterIds include them and the approved text or supplied scene record names that secret as revealed. Reader access or narrative implication alone is not a character exposure. Bind each event to the edition ID, revision and revealing scene. Compare the proposal position with later events: when it moves, removes or changes an earlier exposure, return the exact later event IDs requiring invalidation. Leave unknown recipients and unclear exposure as unresolvedExposureIds rather than guessing who knows.
