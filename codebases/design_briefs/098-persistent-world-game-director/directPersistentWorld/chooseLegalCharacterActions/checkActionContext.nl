---
description: Check that a proposed action can occur at the character's current witnessed location and in the supplied world.
args: { world: PersistentWorldState; utterance: PlayerUtterance; action: LegalAction }
returns: ActionCheck
---
Require action.location to match the player's supplied visibleLocation for this turn. Find the most recent world event for action.actorId by revision; when one exists, its location must also match the proposed action location unless the action itself is a rule-supported movement from that location. If the record does not establish a current location or a movement rule, keep the action unresolved instead of moving the character by narration. Preserve exact event IDs that support the location check.
