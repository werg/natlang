---
description: Choose actions consistent with character knowledge, goals, world rules and current boundaries.
args:{world:PersistentWorldState;utterance:PlayerUtterance;perspective:CharacterPerspective;proposals:LegalAction[];sourceRecord:GameWorld;sourceEvidence:EvidenceItem[]}
returns:ActionSelection
---
For each proposed action, call validateActionEnvelope(world, utterance, proposal), checkActionKnowledge(perspective, proposal), checkActionContext(world, utterance, proposal), and checkActionBoundary(sourceRecord, sourceEvidence, utterance, proposal). Retain the exact ActionCheck results in proposal order. Accept an action only when all four checks are eligible. Preserve a specific blocker and evidence IDs for each rejected action; a failed candidate does not erase another independent eligible action. If every candidate is rejected, return no actions and keep the clarification needed to make progress. Never expose hidden utility or let narrative convenience override simulator constraints.
