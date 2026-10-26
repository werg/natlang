---
description: Build a character-only perspective from authorized witnessed events and memory.
args:{world:PersistentWorldState;characterId:string;actions:LegalAction[]}
returns:CharacterPerspective
---
Combine the character's known fact IDs, memory, witnessed event history and messages actually received. Keep rumor or uncertain account as belief with provenance, never as world truth. Exclude every secret lacking a permitted information channel. List available actions only from this scoped perspective and current location.
