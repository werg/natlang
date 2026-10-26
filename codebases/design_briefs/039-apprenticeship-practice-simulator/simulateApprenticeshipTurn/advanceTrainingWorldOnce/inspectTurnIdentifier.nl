---
args: { action: TypedTrainingAction, checkpoint: PracticeCheckpoint }
returns: TurnReplayCheck
---
Search appliedTransitionIds for this exact action.turnId. If present, recover the matching stored transition and prohibit a second resource mutation. If absent, allow one simulation attempt while preserving the id for retry recovery.
