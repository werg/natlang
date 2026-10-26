---
description: Choose actions consistent with character knowledge, goals, world rules and current boundaries.
args:{world:PersistentWorldState;utterance:PlayerUtterance;perspective:CharacterPerspective;proposals:LegalAction[]}
returns:LegalAction[]
---
Filter proposals against exact legal action schema, character's known context, location, resources and content boundaries. Do not expose hidden utility, reveal unknown facts, or override simulator constraints for narrative convenience. Return no action when nothing is legal; preserve clarifying question.
