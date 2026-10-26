---
args: { turn: ApprenticeshipTurn }
returns: TurnReplayCheck
---
1. Search turn.checkpoint.turnReceipts or an empty list for records whose binding.turnId equals turn.turnId. If exactly one exists, compare its full binding with the current turn's session, source scenario, utterance, source evidence IDs/versions, authority status, actor, world, procedure, scenario family, objective, seed, and action history. Exclude only checkpoint history from this comparison because a valid retry carries the later checkpoint. Return status replay with that exact receipt only when every bound field agrees; otherwise return conflict with no receipt.
2. If more than one receipt has the turn ID, return conflict. If no receipt exists but turn.checkpoint.appliedTransitionIds or turn.checkpoint.seenChallengeIds contains turn.turnId, or turn.checkpoint.lastTransitionId equals it, return unresolved with a blocker that the legacy checkpoint lacks the original complete turn result. Never reconstruct a receipt from a historical ID alone.
3. Otherwise return new with no receipt. In every status, preserve turn.checkpoint; this check never advances, repairs, or rewinds state.
