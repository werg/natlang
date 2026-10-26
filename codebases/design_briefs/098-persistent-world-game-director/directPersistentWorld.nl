---
description: Process a persistent-world player action with character-scoped knowledge and simulator-bound state changes.
args:{ sourceRecord: GameWorld; sourceEvidence: EvidenceItem[];world:PersistentWorldState;utterance:PlayerUtterance;checkpoint:GameCheckpoint;receipts:WorldTransitionReceipt[]}
returns:GameSessionResult
---
Read `worldId`, `ruleSetId`, `sessionIds`, `directorId` and `contentBoundaryIds` from the top level of the GameWorld record. Keep historical session state and current boundaries as separate versions; never replace the world identifiers or safety constraints with defaults. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

If checkpoint world, revision or rule version differs from the supplied authoritative world, reconcile the checkpoint before interpreting new intent. Bind intent = `interpretPlayerIntent(world, utterance, sourceRecord, sourceEvidence)`; speech describing success is not execution. If required clarification is consequential, return the questions and leave world unchanged.

Bind perspective = `projectCharacterKnowledge(world, utterance.characterId, intent.actions)`. Exclude private facts the character has not witnessed or received through an authorized channel. Bind actions = `chooseLegalCharacterActions(world, utterance, perspective, intent.actions)` using only that character's view, goals and conduct boundaries.

Accept state changes only from exact simulator `receipts` matching world ID, prior revision and `clientActionId`. Bind reconciled = `applyWorldReceipts(world, actions, receipts, checkpoint)`; duplicate action IDs return their prior outcome, and unknown receipts remain unknown pending readback. Never narrate unreceipted physical outcomes. Bind continuity = `updateBeliefsAndContinuity(world, reconciled, checkpoint)` from witnessed event facts only. Apply boundary revisions before proposing future content. Rule migrations occur only at a checkpoint and keep prior events replayable.
